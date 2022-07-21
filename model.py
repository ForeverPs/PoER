import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models.resnet import resnet18, resnet34, resnet50


class ResNetCls(nn.Module):
    def __init__(self, num_classes, depth, pretrained=True, dropout=0., latent_dim=128, num_prototypes=2):
        super(ResNetCls, self).__init__()
        if depth == 18:
            model = resnet18(pretrained=pretrained)
        elif depth == 34:
            model = resnet34(pretrained=pretrained)
        else:
            model = resnet50(pretrained=pretrained)
        
        in_channel = model.fc.in_features
        self.backbone = model
        del self.backbone.fc

        self.cls_head = nn.Sequential(
            self.cls_block(in_channel, 256, dropout),
            nn.Linear(256, latent_dim),
            nn.InstanceNorm1d(latent_dim))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.prototypes = nn.Embedding(num_classes * num_prototypes, latent_dim)
        self.num_classes = num_classes
    
    def cls_block(self, channel_in, channel_out, p):
        block = nn.Sequential(
            nn.Linear(channel_in, channel_out),
            nn.ReLU(),
            nn.Dropout(p),
        )
        return block
    

    def get_logits(self, x, y):
        logits = -1.0 * torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
        logits = rearrange(logits, 'b (c p) -> b c p', c=self.num_classes)
        logits, _ = torch.max(logits, dim=-1)
        return logits


    def forward(self, x):
        feats = list()
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feats.append(self.pool(feat).flatten(1))

        feat = self.backbone.layer1(feat)
        feats.append(self.pool(feat).flatten(1))

        feat = self.backbone.layer2(feat)
        feats.append(self.pool(feat).flatten(1))

        feat = self.backbone.layer3(feat)
        feats.append(self.pool(feat).flatten(1))

        feat = self.backbone.layer4(feat)
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feats.append(feat)

        feat = self.cls_head(feat)
        feats.append(feat)
        logits = self.get_logits(feat, self.prototypes.weight)
        return feats, logits