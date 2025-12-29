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


class CoDA(nn.Module):
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
        assert kernel_size % 2 == 1, "kernel_size // 2 == 1"
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(8, in_channels // reduction)
        self.alpha = alpha
        self.use_frequency_gate = use_frequency_gate
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )
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
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=True)
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
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * layerscale_init)
        self.dropout = nn.Dropout(p=0.05)

    def _sigmoid_temp(self, x):
        return torch.sigmoid(x / (self.tau.abs() + 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_h = F.adaptive_avg_pool2d(x, (H, 1)) 
        x_w = F.adaptive_avg_pool2d(x, (1, W))  
        x_w = x_w.permute(0, 1, 3, 2)  
        y = torch.cat([x_h, x_w], dim=2)
        y = self.shared_conv(y)  
        y_h, y_w = torch.split(y, [H, W], dim=2)  
        y_w = y_w.permute(0, 1, 3, 2)
        y_h = self.dw_h(y_h)  
        y_w = self.dw_w(y_w)
        a_h_raw = self.conv_h(y_h) 
        a_w_raw = self.conv_w(y_w) 
        gate_h_from_w = a_w_raw.mean(dim=(2, 3), keepdim=True) 
        gate_w_from_h = a_h_raw.mean(dim=(2, 3), keepdim=True) 
        a_h = self._sigmoid_temp(a_h_raw) * (1.0 + gate_h_from_w)
        a_w = self._sigmoid_temp(a_w_raw) * (1.0 + gate_w_from_h) 
        x_hw = self.dw_hw(x)
        a_hw = self._sigmoid_temp(x_hw)
        x_p = self.dw_p(x)
        a_p = self._sigmoid_temp(x_p)
        if self.use_frequency_gate:
            Xf = torch.fft.fft2(x, norm='ortho') 
            mag = torch.abs(Xf)  
            mag_mean = mag.mean(dim=(2, 3), keepdim=True)
            f_gate = self.freq_gate(mag_mean)  
        else:
            f_gate = 0.0
        attn = a_h + a_w + a_p + a_hw
        attn = attn * (1.0 + f_gate) 
        attn = self.dropout(attn)
        out = x + self.gamma * (x * attn)
        return out


class FFTConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm='backward'): 
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


class DWConv(nn.Module):
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


class FDConv(nn.Module): 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.dwConv = DWConv(in_channels, out_channels)
        self.fftConv = FFTConv(in_channels, out_channels)

    def forward(self, x):
        return self.dwConv(x) + self.fftConv(x)


class HWDownsampling(nn.Module):
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


class Down(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.sfd = FDConv(in_channels, in_channels)
        self.down = HWDownsampling(in_channels, out_channels)

    def forward(self, x):
        context = self.sfd(x)
        out, lh, hl, hh = self.down(context)
        detail = torch.cat([lh, hl, hh], dim=1)
        return out, context, detail


class RefineModule(nn.Module):
    def __init__(self, detail_c, context_c, refine_c):
        super().__init__()
        self.detail_conv = nn.Conv2d(in_channels=detail_c, out_channels=refine_c, kernel_size=1)
        self.context_down = nn.Conv2d(in_channels=context_c, out_channels=refine_c, kernel_size=3, stride=2, padding=1, bias=False)
        self.context_bn = nn.BatchNorm2d(refine_c)
        self.context_relu = nn.ReLU(inplace=True)
        self.attention_gen = nn.Sequential(
            nn.Conv2d(in_channels=refine_c, out_channels=refine_c, kernel_size=1),
            nn.Sigmoid()
        )
        self.output_conv = nn.Conv2d(in_channels=refine_c, out_channels=detail_c, kernel_size=1)

    def forward(self, details, context):
        details_feat = self.detail_conv(details) 
        context_small = self.context_relu(self.context_bn(self.context_down(context))) 
        attn_mask = self.attention_gen(context_small)
        refined_feat = details_feat * attn_mask
        refined_feat = refined_feat + details_feat
        out_details = self.output_conv(refined_feat)  
        return out_details


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convs(x)


class Up(nn.Module):
    def __init__(self, in_c, skip_ct_c, skip_dt_c, out_c):
        super(Up, self).__init__()
        self.iwt = DWTInverse(wave='haar', mode='zero')
        self.ll_prep_conv = nn.Conv2d(in_c, skip_dt_c // 3, 1)
        self.refine = RefineModule(detail_c=skip_dt_c, context_c=skip_ct_c, refine_c=skip_ct_c)
        self.final_conv_block = ConvBlock((skip_dt_c // 3) + skip_ct_c, out_c)

    def forward(self, pre, ct, dt):
        ll = self.ll_prep_conv(pre)
        details_refined_concat = self.refine(dt, ct)
        b, c_full, h_d, w_d = details_refined_concat.shape
        c = c_full // 3
        details_refined_list = [details_refined_concat.reshape(b, c, 3, h_d, w_d)]
        f_recon = self.iwt((ll, details_refined_list))
        f_concat = torch.cat([f_recon, ct], dim=1)
        f_out = self.final_conv_block(f_concat)
        return f_out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@MODELS.register_module()
class SFWnet(nn.Module):
    def __init__(self, bilinear=False):
        super(SFWnet, self).__init__()
        self.inc = OutConv(3, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.att = CoDA(512, kernel_size=9) 
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
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []
    with torch.no_grad():
        for _ in range(iters):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize() 
            curr_time = starter.elapsed_time(ender)
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
    print(f"{flops.total() / 1e9:.3f} GFLOPs")
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
