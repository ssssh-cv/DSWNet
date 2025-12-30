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


"""
The complete code will be provided after the paper is accepted.
"""

class CoDA(nn.Module):   ## HDAM
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
        pass

    def _sigmoid_temp(self, x):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       pass

class FFTConv(nn.Module):  ## DDFB fre branch
    def __init__(self, in_channels, out_channels, mid_channels=None, norm='backward'): 
        super(FFTConv, self).__init__()
        pass

    def forward(self, x):
        pass


class DWConv(nn.Module):  ## DDFB spa branch
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        pass

    def forward(self, x):
        pass


class FDConv(nn.Module):   ## DDFB
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        pass

    def forward(self, x):
        pass

class HWDownsampling(nn.Module):   ## DWT
    def __init__(self, in_channels, out_channels):
        super(HWDownsampling, self).__init__()
        pass

    def forward(self, x):
        pass


class Down(nn.Module):    ## WEM
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        pass

    def forward(self, x):
        pass


class RefineModule(nn.Module):   ## CGDRB
    def __init__(self, detail_c, context_c, refine_c):
        super().__init__()
        pass

    def forward(self, details, context):
        pass


class ConvBlock(nn.Module):  ## BCB
    def __init__(self, in_c, out_c):
        super().__init__()
        pass

    def forward(self, x):
        pass


class Up(nn.Module):   ## DSWRM
    def __init__(self, in_c, skip_ct_c, skip_dt_c, out_c):
        super(Up, self).__init__()
        pass

    def forward(self, pre, ct, dt):
        pass


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(self, x):
        pass


@MODELS.register_module()
class SFWnet(nn.Module):    ## DSW-Net
    def __init__(self, bilinear=False):
        super(SFWnet, self).__init__()
        pass

    def forward(self, x):
        pass


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
