## PixelFormer: Attention Attention Everywhere: Monocular Depth Prediction with Skip Attention
**Disclaimer:** This is NOT the official implementation of PixelFormer. To access the official one, please click [here](https://github.com/ashutosh1807/PixelFormer).

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

## Citation

If you find our work useful in your research, please cite the following:
```bibtex
@InProceedings{Agarwal_2023_WACV,
    author    = {Agarwal, Ashutosh and Arora, Chetan},
    title     = {Attention Attention Everywhere: Monocular Depth Prediction With Skip Attention},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {5861-5870}
}
```

## Contact
For questions about our paper or code, please contact ([@ashutosh1807](https://github.com/ashutosh1807)) or raise an issue on GitHub.



### Acknowledgements
Most of the code has been adpated from CVPR 2022 paper [NewCRFS](https://github.com/aliyun/NeWCRFs). We thank Weihao Yuan for releasing the source code for the same.

Also, thanks to Microsoft Research Asia for opening source of the excellent work [Swin Transformer](https://github.com/microsoft/Swin-Transformer).
