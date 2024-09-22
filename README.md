# MECNet: Multi-Scale Exposure-Consistency Learning via Fourier Transform for Exposure Correction (SMC2024)

![](https://github.com/thisisqiaoqiao/MECNet/blob/main/MECNET_MSEC/img/img.png)

## 1、Parameters setting
```Python
 options/train/train_Enhance.yml
```
## 2、Dataset
MSEC Dataset：https://github.com/mahmoudnafifi/Exposure_Correction
SICE Dataset：https://share.weiyun.com/C2aJ1Cti

## 3、Dataset Preparation
```Python
 python create_txt.py
```

## 4、Training
```Python
 python train.py --opt options/train/train_Enhance.yml
```

## 5、Inference
```Python
 python train.py --opt options/train/train_Enhance.yml
```

## Ours Results
| Expert A | Expert B | Expert C | Expert D | Expert E | Expert E |
|  ----  | ----  |----  |----  |----  |----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |
