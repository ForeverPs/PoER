import torch
import torch.nn as nn
from models.utils import BitLayer
from models.decoder import Decoder
from models.resnet import ResNet18, ResNet34


class PoER(nn.Module):
    def __init__(self, in_channel, num_classes, dropout=0.1, bit=8, latent_dim=32):
        super(PoER, self).__init__()
        self.backbone = ResNet18(in_channel)
        channel_in = self.backbone.out_plane
        self.bitlayer = BitLayer(bit)
    
        self.cls_head = nn.Sequential(
            self.cls_block(channel_in, 256, dropout),
            self.cls_block(256, 128, dropout),
            nn.Linear(128, num_classes))
        
        self.ranking_head = nn.Sequential(
            self.cls_block(channel_in, 256, dropout),
            self.cls_block(256, 128, dropout),
            nn.Linear(128, latent_dim))
        
        self.recon_head = Decoder(output_dim=in_channel, hidden_dim=channel_in)

    def cls_block(self, channel_in, channel_out, p):
        block = nn.Sequential(
            nn.Linear(channel_in, channel_out),
            nn.GELU(),
            nn.Dropout(p),
            nn.LayerNorm(channel_out),
        )
        return block
    
    def forward(self, x):
        feat = self.backbone(x).reshape(x.shape[0], -1)
        feat = self.bitlayer(feat)
        ranking = self.ranking_head(feat)
        cls = self.cls_head(feat)
        recon = self.recon_head(feat)
        return ranking, cls, recon
    

if __name__ == '__main__':
    model = PoER(in_channel=3, num_classes=10)
    x = torch.rand(10, 3, 224, 224)
    ranking, cls, recon = model(x)
    print(ranking.shape, cls.shape, recon.shape)

