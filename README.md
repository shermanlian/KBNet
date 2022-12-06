# Kernel-Aware Burst Blind Super-Resolution, WACV 2023 [paper](https://arxiv.org/abs/2112.07315)

## Dependencies
- OS: Ubuntu 18.04
- Python: Python 3.7
- nvidia :
   - cuda: 10.1
   - cudnn: 7.6.1
- Other reference requirements

## Quick Start
1.Create a conda virtual environment and activate it
```python3
conda create -n pytorch_1.6 python=3.7
source activate pytorch_1.6
```
2.Install PyTorch and torchvision following the official instructions
```python3
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```
3.Install build requirements
```python3
pip3 install -r requirements.txt
```
4.Install DCN
```python3
cd DCNv2-pytorch_1.6
python3 setup.py build develop # build
python3 test.py # run examples and check
```
## Training
```python3
# Modify the root path of training dataset and model etc.
python main.py --n_GPUs 4 --lr 0.0002 --decay 100-200 --save kbnet --model KBNet --n_feats 128 --n_resblocks 8 --n_resgroups 5 --batch_size 32 --burst_size 8 --patch_size 256 --scale 4 --loss 1*L1
```
## Testing
1. Data preparation
```python3
# Modify the output path of test dataset in make_validation_set.py - root = '../test_set_3.2_4.8'
python make_validation_set.py
```
2. Run models on test dataset:
```python3
# Modify the path of test dataset and the path of the trained model
python test.py --root ../test_set_0_1.6 --n_GPUs 1 --model KBNet --n_feats 128 --n_resblocks 8 --n_resgroups 5 --batch_size 64 --burst_size 2 --scale 4 --pre_train ../train_log/KBNet/real_models/kbnet/KBNetbest_epoch.pth
```

## Citations
If EBSR helps your research or work, please consider citing EBSR.
The following is a BibTeX reference.

```
@article{lian2021kernel,
  title={Kernel-aware Raw Burst Blind Super-Resolution},
  author={Lian, Wenyi and Peng, Shanglian},
  journal={arXiv preprint arXiv:2112.07315},
  year={2021}
}
```

## Acknowledgments
This repo is based on [EBSR](https://github.com/Algolzw/EBSR) and [IKC](https://github.com/yuanjunchai/IKC).



