import torch
import torch.nn as nn
from torch.nn import Parameter


# bit operation
class Bitflow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b):
        scale = 1 / 2 ** b
        out = torch.quantize_per_tensor(x, scale=scale, zero_point=0, dtype=torch.quint8).dequantize()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# bit layer
class BitLayer(nn.Module):
    def __init__(self, bit):
        super(BitLayer, self).__init__()
        self.bit = bit

    def forward(self, x):
        out = Bitflow.apply(x, self.bit)
        return out


# Position Attention Module
class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        down_scale_dim = max(in_dim // 8, 3)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=down_scale_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=down_scale_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out
        return out


# Channel Attention Module
class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out
        return out


if __name__ == '__main__':
    channel = 512
    pam = PAM_Module(in_dim=channel)
    cam = CAM_Module(in_dim=channel)

    feature_map = torch.rand(10, channel, 7, 7)

    pam_out = pam(feature_map)
    cam_out = cam(feature_map)

    print(pam_out.shape, cam_out.shape)