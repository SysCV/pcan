## Installation

### Requirements
- Linux
- Python 3.6+ 
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- NCCL 2+
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

### Install PCAN

a. Create a conda virtual environment and activate it.
```shell
conda create -n pcan python=3.7 -y
conda activate pcan
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

c. Install mmcv and mmdetection.

```shell
pip install mmcv-full==1.2.7
pip install mmdet==2.10.0
```

You can also refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md).

Note that mmdetection uses their forked version of pycocotools via the github repo instead of pypi for better compatibility. If you meet issues, you may need to re-install the cocoapi through
```shell
pip uninstall pycocotools
pip install mmpycocotools
```

d. Install mot metrics
```shell
pip install motmetrics
```

e. Install PCAN
```shell
python setup.py develop
```
