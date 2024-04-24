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
2) Download [CityScapes](https://www.cityscapes-dataset.com/) into datasets directory

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
- [ ] Inception v2,3
- [x] ResNext
- [x] DenseNet
- [ ] Inception v4
- [ ] SqueezeNet
- [ ] WRN
- [x] Xception
- [ ] DPN
- [ ] MobileNet v1
- [ ] RAN
- [ ] ShuffleNet
- [ ] SENet
- [ ] MobileNet v2
- [ ] EfficientNet
- [ ] ViT
- [ ] CvT
- [ ] Swin Transformer
