# Domain Decorrelation with Potential Energy Ranking
[![NICO](https://img.shields.io/badge/2022%20ECCV%20Workshop-Jury%20Award-FFD500?style=flat&labelColor=005BBB)](https://codalab.lisn.upsaclay.fr/competitions/4083)

[![NICO](https://img.shields.io/badge/2023%20AAAI%20-Main%20Track-FFD500?style=flat&labelColor=005BBB)](https://codalab.lisn.upsaclay.fr/competitions/4083)

- [Accepted as AAAI 2023](https://arxiv.org/abs/2207.12194)

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
<img src="https://github.com/ForeverPs/PoER/raw/master/data/pacs_result.png" width="500px"/>

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
      author={Sen Pei and Jiaxi Sun and Yida Xu and and Shiming Xiang and Gaofeng Meng},
      year={2022},
      eprint={2207.12194},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
