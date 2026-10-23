import pytorch_wavelets
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath
from mmengine.utils import is_tuple_of
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
from mmagic.registry import MODELS


class CoDA(nn.Module): ## HDAM

    def __init__(
            self,
            in_channels: int,
            reduction: int = 16,
            alpha: float = 0.9,
            kernel_size: int = 7,
            use_frequency_gate: bool = True,
            layerscale_init: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 需为奇数，便于对齐（padding 对称）"
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(8, in_channels // reduction)  # 瓶颈通道数
        self.alpha = alpha
        self.use_frequency_gate = use_frequency_gate

        # 共享 MLP：对 H/W 拼接后的方向统计做通道压缩与非线性映射
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            # nn.InstanceNorm2d(self.mid_channels),
            # nn.LeakyReLU(0.1, inplace=False)
        )

        # 方向长程建模：深度可分一维卷积（沿 H 与 W）
        # 目的：对每个方向引入更强的跨行/跨列上下文
        pad = kernel_size // 2
        self.dw_h = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            groups=self.mid_channels,
            bias=True,
        )
        self.dw_w = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            groups=self.mid_channels,
            bias=True,
        )

        # 方向映射回原通道维度
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=True)

        # 频域门控：逐通道门（0~1），由幅度谱的全局统计产生
        if self.use_frequency_gate:
            self.freq_gate = nn.Sequential(
                nn.Conv2d(in_channels, max(8, in_channels // reduction), kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(8, in_channels // reduction), in_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

        self.dw_hw = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, padding=pad,
                               groups=self.mid_channels, bias=True)
        self.dw_p = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0,
                              groups=self.mid_channels, bias=True)

        # 可学习温度，用于调节 Sigmoid 的锐度（τ 越小越“硬”）
        self.tau = nn.Parameter(torch.tensor(1.0))

        # 层缩放参数（LayerScale），稳定深层残差叠加
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * layerscale_init)

        # 轻微随机失活，进一步抑制过拟合（可选）
        self.dropout = nn.Dropout(p=0.05)

    def _sigmoid_temp(self, x):
        # 带温度的 Sigmoid：sigmoid(x / tau)
        return torch.sigmoid(x / (self.tau.abs() + 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, C, H, W]
        输出：
            y: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # -------- 1) 方向统计（与 ACA 一致的骨架）--------
        x_h = F.adaptive_avg_pool2d(x, (H, 1))  # [B, C, H, 1] —— 沿宽度聚合，保留行方向
        x_w = F.adaptive_avg_pool2d(x, (1, W))  # [B, C, 1, W] —— 沿高度聚合，保留列方向
        x_w = x_w.permute(0, 1, 3, 2)  # 变成 [B, C, W, 1]，便于与 H 方向拼接

        # 拼接后用共享 MLP 抽取方向相关的通道表征
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.shared_conv(y)  # [B, C', H+W, 1]，C' = mid_channels

        # 切分回两个方向分支，并做“方向深度可分一维卷积”的长程建模
        y_h, y_w = torch.split(y, [H, W], dim=2)  # y_h:[B,C',H,1], y_w:[B,C',W,1]
        y_w = y_w.permute(0, 1, 3, 2)  # y_w:[B,C',1,W]

        y_h = self.dw_h(y_h)  # 在 H 方向做 k×1 的 DWConv
        y_w = self.dw_w(y_w)  # 在 W 方向做 1×k 的 DWConv

        # 映射回原通道数
        a_h_raw = self.conv_h(y_h)  # [B,C,H,1]
        a_w_raw = self.conv_w(y_w)  # [B,C,1,W]

        # -------- 2) 跨门控融合（Cross-gating）--------
        # 使用对向分支的全局门对本分支进行调制，增强“相互选择/抑制”
        gate_h_from_w = a_w_raw.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        gate_w_from_h = a_h_raw.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]

        a_h = self._sigmoid_temp(a_h_raw) * (1.0 + gate_h_from_w)  # [B,C,H,1]
        a_w = self._sigmoid_temp(a_w_raw) * (1.0 + gate_w_from_h)  # [B,C,1,W]

        x_hw = self.dw_hw(x)
        a_hw = self._sigmoid_temp(x_hw)
        x_p = self.dw_p(x)
        a_p = self._sigmoid_temp(x_p)

        # -------- 3) 频域门控（Dual-domain 之二：Frequency）--------
        if self.use_frequency_gate:
            # 计算幅度谱的均值特征（逐通道）
            # 注：仅用于产生门控标量，不做反变换，计算量可控
            Xf = torch.fft.fft2(x, norm='ortho')  # 复数张量
            # amp = torch.angle(Xf)
            mag = torch.abs(Xf)  # 幅度谱 [B,C,H,W]
            mag_mean = mag.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
            # amp_mean = amp.mean(dim=(2, 3), keepdim=True)
            # f_gate = self.freq_gate(amp_mean)
            f_gate = self.freq_gate(mag_mean)  # [B,C,1,1] in [0,1]
        else:
            f_gate = 0.0

        # -------- 4) 融合与输出 --------
        # 方向注意力相加（与坐标注意力一致的融合策略），再乘以频域门控（1 + f_gate）
        # attn = a_h + a_w  # [B,C,H,W]（自动广播）
        attn = a_h + a_w + a_p + a_hw
        attn = attn * (1.0 + f_gate)  # 双域融合：空间方向 + 频域门控
        attn = self.dropout(attn)

        # 残差式输出，LayerScale 提升稳定性
        out = x + self.gamma * (x * attn)
        return out


class FFTConv(nn.Module):  ## DDFB fre branch
    def __init__(self, in_channels, out_channels, mid_channels=None, norm='backward'):  # 'ortho'
        super(FFTConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dim = 1
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = self.dim
        y = torch.fft.rfft2(x, norm=self.norm)
        mag = torch.abs(y)
        pha = torch.angle(y)
        mag = self.conv1(mag)
        pha = self.conv2(pha)
        y_real = mag * torch.cos(pha)
        y_imag = mag * torch.sin(pha)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class DWConv(nn.Module):  ## DDFB spa branch
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.depth_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channels)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
        self.In = nn.InstanceNorm2d(in_channels)
        self.act = nn.LeakyReLU(0.1, inplace=False)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              groups=1)
        self.dw3conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=in_channels)
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x1 = self.depth_conv(x)
        x1 = self.In(x1)
        x1 = self.act(x1)
        x1 = self.conv(x1)
        x1 = x + x1
        x2 = self.dw3conv(x1)
        x2 = self.In(x2)
        x2 = self.act(x2)
        out = self.point_conv(x2)
        return out


class FDConv(nn.Module):  ## DDFB
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.dwConv = DWConv(in_channels, out_channels)
        self.fftConv = FFTConv(in_channels, out_channels)
        # self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # spa = self.dwConv(x)
        # fre = self.fftConv(x)
        # out = torch.cat([spa, fre], dim=1)
        # out = self.fuse(out)
        # return out
        return self.dwConv(x) + self.fftConv(x)


class HWDownsampling(nn.Module):  ## DWT
    def __init__(self, in_channels, out_channels):
        super(HWDownsampling, self).__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.change = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        yL = self.change(yL)
        return yL, y_LH, y_HL, y_HH


class Down(nn.Module):  ## WEM
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.sfd = FDConv(in_channels, in_channels)
        self.down = HWDownsampling(in_channels, out_channels)

    def forward(self, x):
        context = self.sfd(x)
        out, lh, hl, hh = self.down(context)
        detail = torch.cat([lh, hl, hh], dim=1)
        return out, context, detail


class RefineModule(nn.Module):  ## CGDRB
    """
    细节精炼模块。
    使用高分辨率的 Skip_Context 来指导(门控)低分辨率的 Skip_Detail。
    """

    def __init__(self, detail_c, context_c, refine_c):
        """
        参数:
            detail_c (int): Skip_Detail 的通道数 (例如 3*C_in)
            context_c (int): Skip_Context 的通道数 (例如 C_out)
            refine_c (int): 内部处理的通道数
        """
        super().__init__()
        # 1. 准备 Detail: (B, 3C, H/2, W/2) -> (B, refine_c, H/2, W/2)
        self.detail_conv = nn.Conv2d(in_channels=detail_c, out_channels=refine_c, kernel_size=1)

        # 2. 准备 Context: (B, C_ctx, H, W) -> (B, refine_c, H/2, W/2)
        self.context_down = nn.Conv2d(in_channels=context_c, out_channels=refine_c, kernel_size=3, stride=2, padding=1,
                                      bias=False)
        self.context_bn = nn.BatchNorm2d(refine_c)
        self.context_relu = nn.ReLU(inplace=True)
        # self.context_bn = nn.InstanceNorm2d(refine_c)
        # self.context_relu = nn.LeakyReLU(0.1, inplace=False)

        # 3. 生成注意力图
        self.attention_gen = nn.Sequential(
            nn.Conv2d(in_channels=refine_c, out_channels=refine_c, kernel_size=1),
            nn.Sigmoid()
        )

        # 4. 残差连接和输出投影
        self.output_conv = nn.Conv2d(in_channels=refine_c, out_channels=detail_c, kernel_size=1)

    def forward(self, details, context):
        """
        details: Skip_Detail (B, 3C_in, H/2, W/2)
        context: Skip_Context (B, C_out, H, W)
        """
        # 1. 编码 Detail
        details_feat = self.detail_conv(details)  # (B, refine_c, H/2, W/2)

        # 2. 下采样并编码 Context
        context_small = self.context_relu(self.context_bn(self.context_down(context)))  # (B, refine_c, H/2, W/2)

        # 3. 生成注意力图
        # 上下文信息现在判断哪些特征是重要的
        attn_mask = self.attention_gen(context_small)

        # 4. 门控（精炼）
        # 使用上下文的判断 (attn_mask) 去筛选原始细节 (details_feat)
        refined_feat = details_feat * attn_mask

        # 5. 残差连接 (让网络学习修正，而不是完全替换)
        refined_feat = refined_feat + details_feat

        # 6. 投影回原始 Detail 通道数
        out_details = self.output_conv(refined_feat)  # (B, 3C_in, H/2, W/2)

        return out_details


class ConvBlock(nn.Module):   ## BCB
    def __init__(self, in_c, out_c):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            # nn.InstanceNorm2d(out_c),
            # nn.LeakyReLU(0.1, inplace=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            # nn.InstanceNorm2d(out_c),
            # nn.LeakyReLU(0.1, inplace=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convs(x)


class Up(nn.Module):   ## DSWRM
    def __init__(self, in_c, skip_ct_c, skip_dt_c, out_c):
        super(Up, self).__init__()
        self.iwt = DWTInverse(wave='haar', mode='zero')
        self.ll_prep_conv = nn.Conv2d(in_c, skip_dt_c // 3, 1)
        # 细节精炼模块
        self.refine = RefineModule(detail_c=skip_dt_c, context_c=skip_ct_c, refine_c=skip_ct_c)
        # 最终融合 (Concat(F_recon, Skip_Context))
        # F_recon 的通道数 = skip_det_c // 3
        # Skip_Context 的通道数 = skip_ctx_c
        self.final_conv_block = ConvBlock((skip_dt_c // 3) + skip_ct_c, out_c)

    def forward(self, pre, ct, dt):
        ll = self.ll_prep_conv(pre)
        details_refined_concat = self.refine(dt, ct)
        # 3. 准备 i-DWT
        # 将 (B, 3C, H/2, W/2) 转换回 (B, C, 3, H/2, W/2)
        b, c_full, h_d, w_d = details_refined_concat.shape
        c = c_full // 3
        details_refined_list = [details_refined_concat.reshape(b, c, 3, h_d, w_d)]

        # 4. 逆小波重构 (上采样)
        # i-DWT ( (B,C,H/2,W/2), [(B,C,3,H/2,W/2)] ) -> (B, C, H, W)
        f_recon = self.iwt((ll, details_refined_list))

        # 5. 最终融合
        # 拼接重构图和上下文跳跃
        f_concat = torch.cat([f_recon, ct], dim=1)
        f_out = self.final_conv_block(f_concat)  # (B, C_out, H, W)

        return f_out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@MODELS.register_module()
class SFWnet(nn.Module):    ## DSW-Net

    def __init__(self, bilinear=False):
        super(SFWnet, self).__init__()

        self.inc = OutConv(3, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.att = CoDA(512, kernel_size=9)  # HW=16   k=7/9

        self.up1 = Up(512, 256, 256 * 3, 256)
        self.up2 = Up(256, 128, 128 * 3, 128)
        self.up3 = Up(128, 64, 64 * 3, 64)
        self.up4 = Up(64, 32, 32 * 3, 32)
        self.outc = OutConv(32, 3)

    def forward(self, x):
        x1 = self.inc(x)

        ot1, ct1, dt1 = self.down1(x1)
        ot2, ct2, dt2 = self.down2(ot1)
        ot3, ct3, dt3 = self.down3(ot2)
        ot4, ct4, dt4 = self.down4(ot3)

        attn = self.att(ot4)

        u1 = self.up1(attn, ct4, dt4)
        u2 = self.up2(u1, ct3, dt3)
        u3 = self.up3(u2, ct2, dt2)
        u4 = self.up4(u3, ct1, dt1)

        logits = self.outc(u4)

        return logits


def measure_inference_time(model, device='cuda', img_size=(1, 3, 256, 256),
                           warmup=50, iters=200):
    model.eval()
    dummy_input = torch.randn(img_size).to(device)

    # --- Warmup ---
    # CUDA 第一次运行会有编译、cache 创建等开销，因此需要热身
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # --- True Timing ---
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(iters):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU完成
            curr_time = starter.elapsed_time(ender)  # ms
            timings.append(curr_time)

    avg_ms = sum(timings) / len(timings)
    return avg_ms


if __name__ == '__main__':
    from thop import profile, clever_format

    t = torch.randn(1, 3, 256, 256).cuda()
    model = SFWnet().cuda()
    macs, params = profile(model, inputs=(t,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    device = 'cuda'
    model = SFWnet().to(device)
    model.eval()

    x = torch.randn(1, 3, 256, 256).to(device)

    flops = FlopCountAnalysis(model, x)

    print("=== Params ===")
    print(parameter_count_table(model))

    print("\n=== FLOPs ===")
    print(f"{flops.total() / 1e9:.3f} GFLOPs")  # 注意：FLOPs=MACs*2

    import time

    avg_time = measure_inference_time(
        model,
        device=device,
        img_size=(1, 3, 256, 256),
        warmup=50,
        iters=200
    )

    print(f"\n=== Inference Time ===")
    print(f"Average latency: {avg_time:.3f} ms")
    print(f"FPS: {1000.0 / avg_time:.2f}")

