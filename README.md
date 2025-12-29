## Introduction
This project is based on MMagic, an open-source tool from OpenMMLab. For details, please see https://github.com/open-mmlab/mmagic.

Here is the key code:

    .
    ├── ...
    ├── configs
    │   ├── SFW
    │   │   ├── sfw.py
    │   │   ├── test.py
    ├── mmagic
    │   ├── models
    │   │   ├── base_models
    │   │   │   ├── base_uw_model.py
    │   │   ├── editors
    │   │   │   ├── SFW
    │   │   │   │   ├── sfw.py
    │   │   │   │   ├── __init__.py
    │   │   │   ├── __init__.py
    │   │   ├── losses
    │   │   │   ├── perceptual_loss.py
    │   │   │   ├── pixelwise_loss.py
    │   │   │   ├── ssim_loss.py
    │   │   │   ├── __init__.py
    └── ...
    

You just need to put or replace the code I provided in the corresponding folders.


## Data

The dataset can be stored anywhere. During training, only the path in the config file needs to be modified; the same applies to testing.



## Training and Testing

Training and testing commands can be viewed at the MMagic’s documentation "https://mmagic.readthedocs.io/en/latest/".

