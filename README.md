# ArtiSplatfacto

**ArtiSplatfacto** is a custom fine-tuning extension of the [Splatfacto](https://github.com/nerfstudio-project/nerfstudio) method in Nerfstudio, designed for object-specific 3D Gaussian Splatting.

## Features

- Loads and filters 3D Gaussians based on object masks
- Applies pose transformation for fine-tuning on object-only regions
- Optionally retains background (non-object) Gaussians as fixed components
- Seamlessly integrates with Nerfstudio's training pipeline via `nerfstudio-method-template`


## Installation
Before installing Arti-Splatfacto, make sure you have installed Nerfstudio following these [instructions](https://docs.nerf.studio/quickstart/installation.html).
```
conda activate nerfstudio
cd arti-splatfacto
pip install -e .
ns-install-cli
```
## Running the new method

Check the bash script to finetune a pretrained model

