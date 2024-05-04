# Image Classification in PyTorch

This repository contains the implementation of image classification models in PyTorch.

## Prerequisites
Install the required dependencies with the following command:
```
pip install -r requirements.txt
```


## Usage
1) Install the required dependencies with the following command:
```
pip install -r requirements.txt
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
- [ ] SqueezeNet
- [x] Wide Residual Networks
- [x] Xception
- [x] Dual Path Networks
- [ ] MobileNet v1
- [x] Residual Attention Network
- [ ] ShuffleNet
- [x] SE-ResNet
- [x] CBAM-ResNet
- [ ] MobileNet v2
- [ ] EfficientNet
- [ ] Vision Transformer
- [ ] Convolutional vision Transformer
- [ ] Swin Transformer
