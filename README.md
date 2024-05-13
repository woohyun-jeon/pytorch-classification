# Image Classification in PyTorch

This repository contains the implementation of image classification models in PyTorch.

## Prerequisites
* python >= 3.6
* torch >= 1.8.1
* torchvision >= 0.9.1


## Usage
1) Clone the repository and install the required dependencies with the following command:
```
$ git clone https://github.com/woohyun-jeon/pytorch-classification.git
$ cd pytorch-classification
$ pip install -r requirements.txt
```
2) Download [ImageNet](https://image-net.org/) into datasets directory

The directory structure should be as follows:
```
  datasets/
    ILSVRC/      
      Annotations/
        CLS-LOC/
            train/
                n01440764/
                    n01440764_10040.JPEG
                    ...
                ...
            val/
                n01440764/
                    n01440764_0000001.JPEG
                ...
      Data/
        CLS-LOC/
            train/
                n01440764/
                    n01440764_10040.xml
                    ...
                ...
            val/
                n01440764/
                    n01440764_0000001.xml
                ...
            test/
                *
                ...
      ImageSets/
        CLS-LOC/
            test.txt
            train_cls.txt
            train_loc.txt
            val.txt      
```

3) Run ```python train.py``` for training

## Supported Models
- [x] Inception v1
- [x] VGGNet
- [x] ResNet
- [x] Inception v2,3
- [x] Pre-Activation ResNet
- [ ] ResNext
- [x] DenseNet
- [ ] Inception v4
- [x] SqueezeNet
- [x] Wide Residual Networks
- [ ] Xception
- [x] Dual Path Networks
- [ ] MobileNet v1
- [ ] MobileNet v2
- [x] Residual Attention Network
- [x] MnasNet
- [x] ShuffleNet v1
- [x] ShuffleNet v2
- [x] SE-ResNet
- [x] CBAM-ResNet
- [x] EfficientNet
- [x] Vision Transformer
- [x] Swin Transformer