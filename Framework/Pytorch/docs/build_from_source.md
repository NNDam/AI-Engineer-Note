# Pytorch

## Build pytorch from source (best config for AMD CPU & NVIDIA-GPU)
We will use OpenBLAS instead of MKL & MKLDNN
```
# Install anaconda (if not)
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
source ~/anaconda3/bin/activate

# Install dependencies
conda create -n myenv_pytorch_1.9 python=3.8
conda activate myenv_pytorch_1.9
conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
pip install ninja

# Build
git clone --recursive --branch v1.9.1  https://github.com/pytorch/pytorch.git
cd pytorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
USE_NCCL=ON USE_CUDNN=OFF USE_CUDA=ON USE_MKL=OFF USE_MKLDNN=OFF python setup.py install
```
## Compatible with
- TorchVision: 0.10.1
- OpenCV: 4.6.0
- MMCV: 1.3.3
- MMCV Compiler: GCC 9.4
- MMCV CUDA Compiler: 11.3
- MMDetection: 2.7.0+e78eee5

