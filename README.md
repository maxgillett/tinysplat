# tinysplat

Tinysplat is a minimal [3D Gaussian splatting](https://arxiv.org/abs/2308.04079) implementation aiming to reach SOTA performance in training speed and accuracy on few-view indoor training tasks. It supports depth-guided regularization in the manner of [Chung et al, 2023](https://arxiv.org/abs/2311.13398), with in-progress support for [Reconfusion](https://arxiv.org/abs/2312.02981)-inspired diffusion model regularization. It currently leverages the [gsplat library](https://github.com/nerfstudio-project/gsplat) developed by those on the Nerfstudio team. Tinysplat was originally written to use [Tinygrad](https://github.com/tinygrad/tinygrad), but has since switched to PyTorch due to poor ergonomics of custom CUDA kernels in Tinygrad.

![Training](/docs/static/training.gif?raw=true)

### Quickstart

1. Prepare a dataset for 3D reconstruction, for example the SfM-processed Tanks and Temples dataset provided by INRIA's FUNGRAPH [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

2. Start the training procedure:

`LOG_LEVEL=DEBUG python scripts/train.py --train --regularize-depth --dataset-dir=datasets/truck`

A full list of the available options can be displayed with `python scripts/train.py --help`.

3. View the scene during training with a freely moving camera by launching the viewer:

`cd viewer`
`npx vite`

You can now navigate to `http://localhost:5173`, using the WASD+QE keys and mouse to explore.
