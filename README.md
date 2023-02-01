# Domain Decorrelation with Potential Energy Ranking
[![NICO](https://img.shields.io/badge/2022%20ECCV%20Workshop-Jury%20Award-FFD500?style=flat&labelColor=005BBB)](https://codalab.lisn.upsaclay.fr/competitions/4083)

[![NICO](https://img.shields.io/badge/2023%20AAAI%20-Main%20Track-FFD500?style=flat&labelColor=005BBB)](https://codalab.lisn.upsaclay.fr/competitions/4083)

- [Accepted by AAAI 2023](https://arxiv.org/abs/2207.12194)

<img src="https://github.com/ForeverPs/PoER/raw/master/data/pipeline.jpg" width="1000px"/>

Official PyTorch Implementation
> Sen Pei, Jiaxi Sun
> <br/> Institute of Automation, Chinese Academy of Sciences

## Datasets
- PACS: photo, art painting, cartoon, sketch
- VLCS: Pascal VOC, LabelMe, Caltech, SUN09
- OfficeHome: Artistic, Clipart, Product, Real World
- Digits-DG: MNIST, MNIST-M, SVHN, SYN
- NICO: 19 classes belonging to 65 domains
- download here: [Baidu Disk (ql17)](https://pan.baidu.com/s/1-_3zqCId87_JXaMyTaeaQw)

## Data Augmentation Scheme
- RandomResizedCrop
- RandomHorizontalFlip
- ColorJitter
- Normalize

## Architectures
- backbone: ResNet-18
- distance-based cross entropy

## Pretrained Models
- ImageNet Pre-trained models

## Training
- `python -m torch.distributed.launch --nproc_per_node=8 pacs_train.py --n_gpus=8`
- `python -m torch.distributed.launch --nproc_per_node=8 vlcs_train.py --n_gpus=8`
- `python -m torch.distributed.launch --nproc_per_node=8 nico_train.py --n_gpus=8`
- `python -m torch.distributed.launch --nproc_per_node=8 digits_train.py --n_gpus=8`
- `python -m torch.distributed.launch --nproc_per_node=8 officehome_train.py --n_gpus=8`

## Metrics

| Methods | Art Painting | Cartoon | Photo | Sketch | Average |
| :---: | :---: | :---: | :---: | :---: | :---: |
| MMD-AAE | 75.20 | 72.70 | 96.00 | 64.20 | 77.03 |
| CCSA | 80.50 | 76.90 | 93.60 | 66.80 | 79.45 |
| ResNet-18 | 77.00 | 75.90 | 96.00 | 69.20 | 79.53 |
| StableNet | 80.16 | 74.15 | 94.24 | 70.10 | 79.66 |
| JiGen | 79.40 | 75.30 | 96.00 | 71.60 | 80.50 |
| CrossCrad | 79.80 | 76.80 | 96.00 | 70.20 | 80.70 |
| DANN | 80.20 | 77.60 | 95.40 | 70.00 | 80.80 |
| Epi-FCR | 82.10 | 77.00 | 93.90 | 73.00 | 81.50 |
| MetaReg | 83.70 | 77.20 | 95.50 | 70.30 | 81.70 |
| GCPL | 82.64 | 75.02 | 96.40 | 73.36 | 81.86 |
| EISNet | 81.89 | 76.44 | 95.93 | 74.33 | 82.15 |
| L2A-OT | 83.30 | 78.20 | 96.20 | 73.60 | 82.83 |
| MixStyle | 84.10 | 78.80 | 96.10 | 75.90 | 83.70 |
| PoER (Ours) | 85.30 | 77.69 | 96.42 | 77.30 | 84.18 |

## Reference
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Deeper, Broader and Artier Domain Generalization](https://openaccess.thecvf.com/content_iccv_2017/html/Li_Deeper_Broader_and_ICCV_2017_paper.html) (ICCV, 2017)
- [Robust classification with convolutional prototype learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_Robust_Classification_With_CVPR_2018_paper.html) (CVPR, 2018)
- [Domain Generalization via Entropy Regularization](https://proceedings.neurips.cc/paper/2020/hash/b98249b38337c5088bbc660d8f872d6a-Abstract.html) (NeurIPS, 2020)

## Citation
>
```
@misc{pei2022domain,
      title={Domain Decorrelation with Potential Energy Ranking}, 
      author={Sen Pei and Jiaxi Sun and Richard Yida Xu and and Shiming Xiang and Gaofeng Meng},
      year={2022},
      eprint={2207.12194},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
