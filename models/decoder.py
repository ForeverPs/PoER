import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim=512):
        super(Decoder, self).__init__()
        linear_block = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
        )
        conv_block = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )

        self.linear_block = linear_block
        self.conv_block = conv_block

        self.head = nn.Conv2d(16, output_dim, kernel_size=3, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm2d(output_dim)

    def forward(self, input):
        output = self.linear_block(input)
        output = output.view(-1, 512, 2, 2)
        output = self.conv_block(output) # [batch, 16, 64, 64]
        output = self.head(output)       # [batch, 3, 32, 32]
        output = self.batch_norm(output)
        output = self.sigmoid(output)
        return output


if __name__ == '__main__':
    model = Decoder(output_dim=3, hidden_dim=512)
    x = torch.rand(64, 512)
    y = model(x)
    print(y.shape)