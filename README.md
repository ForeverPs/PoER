# PoER: Potential Energy Ranking for Out-of-distribution Generalization

<img src="https://github.com/ForeverPs/PoER/raw/master/data/PoER.jpg" width="1000px"/>

## Datasets
- MNIST, fashion-MNIST, Omniglot
- CIFAR-10, CIFAR-100, LSUN, SVHN, TinyImageNet
- Put the original data in `data` folder.

## Data Augmentation Scheme
- To Do

## Architectures
`currently`
- Backbone: ResNet34
- Ranking head and Classification head: MLP
- Reconstruction head: `ConvTranpose2d`
- Quantization: `torch.quantize_per_tensor`

## Pretrained Models
- To Do

## Training
- To Do

## Reference
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
