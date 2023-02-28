## PixelFormer: Attention Attention Everywhere: Monocular Depth Prediction with Skip Attention
**Disclaimer:** This is NOT the official implementation of PixelFormer. Please refer to [the official implementation](https://github.com/ashutosh1807/PixelFormer) regard to further, more-detailed information.

**[Paper](https://arxiv.org/pdf/2210.09071)** <br />


### Installation
```
conda create -n pixelformer python=3.8
conda activate pixelformer
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1
pip install matplotlib tqdm tensorboardX timm mmcv
```


### Datasets
You can prepare the datasets KITTI and NYUv2 according to [here](https://github.com/cleinc/bts), and then modify the data path in the config files to your dataset locations.


### Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training the NYUv2 model:
```
python pixelformer/train.py configs/arguments_train_nyu.txt
```

Training the KITTI model:
```
python pixelformer/train.py configs/arguments_train_kittieigen.txt
```


### Evaluation
Evaluate the NYUv2 model:
```
python pixelformer/eval.py configs/arguments_eval_nyu.txt
```

Evaluate the KITTI model:
```
python pixelformer/eval.py configs/arguments_eval_kittieigen.txt
```

## Pretrained Models
* You can download the pretrained models "nyu.pt" and "kitti.pt" from [here](https://drive.google.com/drive/folders/1Feo67jEbccqa-HojTHG7ljTXOW2yuX-X?usp=share_link).
