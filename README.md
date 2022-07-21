# Domain Decorrelation with Potential Energy Ranking

<img src="https://github.com/ForeverPs/PoER/raw/master/data/PoER.jpg" width="1000px"/>

## Datasets
- PACS: photo, art painting, cartoon, sketch
- put the dataset in `data/PACS/` folder 
- we provide data split in `data/PACS/datalist/`

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
- `python -m torch.distributed.launch --nproc_per_node=8 train.py --n_gpus=8`

## Reference
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Deeper, Broader and Artier Domain Generalization](https://openaccess.thecvf.com/content_iccv_2017/html/Li_Deeper_Broader_and_ICCV_2017_paper.html) (ICCV, 2017)
- [Robust classification with convolutional prototype learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_Robust_Classification_With_CVPR_2018_paper.html) (CVPR, 2018)
- [Domain Generalization via Entropy Regularization](https://proceedings.neurips.cc/paper/2020/hash/b98249b38337c5088bbc660d8f872d6a-Abstract.html) (NeurIPS, 2020)
