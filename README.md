# DSW-Net
A dual-skip connection wavelet network for underwater image enhancement.

This repository includes code for the following paper: 

🚀 **DSW-Net: A dual-skip connection wavelet network for underwater image enhancement**  
**✅ Accepted in Knowledge-Based Systems (KBS)**  
**Authors:** Hui Sun & et al.   
**Paper:** [DSW-Net](https://doi.org/10.1016/j.knosys.2026.116506)
>***Abstract:***  
*Underwater images commonly exhibit severe colour distortion, low contrast and blurred texture details because of the wavelength-selective absorption and scattering of light in water. Existing convolutional neural network-based approaches, particularly those using U-shaped architectures, often suffer from feature entanglement in underwater image enhancement. In particular, conventional pooling operations irreversibly blend low-frequency global structures with high-frequency local details, causing the decoder to compromise texture details while correcting colour casts. To address this limitation, we developed a dual-skip wavelet network (DSW-Net). Integrating the wavelet transform, we built a wavelet encoder module to achieve explicit feature decoupling across frequency bands. We introduced a dual-skip connection mechanism, which combined with a context-guided detail refinement block, leveraged semantic context priors to adaptively suppress noise and enhance fine textures. A hybrid-domain attention module was incorporated to synergistically aggregate spatial- and frequency-domain features, capturing global degradation characteristics effectively. Extensive experiments conducted on multiple benchmark datasets demonstrated that DSW-Net exhibited strong competitiveness in full-reference and non-reference image quality assessments. The proposed method also showed clear advantages in colour correction and detail preservation over existing state-of-the-art approaches.* 
<p align="middle">
  <img src="./arch.png">
</p>
<p align="middle">
  <img src="./90.png">
</p>

## Results of our method
<img src="./results.png" width = "800" height = "820" div align=center />

## Introduction
This project is based on MMagic, an open-source tool from OpenMMLab. For details, please see https://github.com/open-mmlab/mmagic.

Here is the key code:

    .
    ├── ...
    ├── configs
    │   ├── SFW
    │   │   ├── sfw.py
    │   │   ├── test.py
    ├── ...
    ├── mmagic
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── base_models
    │   │   │   ├── base_uw_model.py
    │   │   ├── editors
    │   │   │   ├── SFW
    │   │   │   │   ├── sfw.py (The complete code will be provided after the paper is accepted.)
    │   │   │   │   ├── __init__.py
    │   │   │   ├── __init__.py
    │   │   ├── losses
    │   │   │   ├── perceptual_loss.py
    │   │   │   ├── pixelwise_loss.py
    │   │   │   ├── ssim_loss.py
    │   │   │   ├── __init__.py
    ├── ...
    └── ...
    


## Data

The dataset can be stored anywhere. During training, only the path in the config file needs to be modified; the same applies to testing.



## Training and Testing

Training and testing commands can be viewed at the MMagic’s documentation "https://mmagic.readthedocs.io/en/latest/".

