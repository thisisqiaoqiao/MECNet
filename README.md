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
|        | Expert A | Expert B | Expert C | Expert D | Expert E | Average |
|  ----  |  ----  | ----  |  ----  |  ----  |  ----  |  ----  |
| PSNR |  20.653  | 22.931 | 22.769 | 21.017 | 21.040 | 21.682 |
| SSIM |  0.831   | 0.882  | 0.871  | 0.856  | 0.863  | 0.861  |
