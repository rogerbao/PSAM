import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, mode='add'):
        super(ResidualBlock, self).__init__()
        self.mode = mode
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        if self.mode == 'add':
            return x + self.main(x)
        elif self.mode == 'multi':
            return x * self.main(x) + x


class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze()


class MBGenerator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(MBGenerator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.F = nn.Sequential(*layers)

        # Down-Sampling
        layers = []
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2-1):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        # Multiplier and Bias
        layers = []
        layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.M = nn.Sequential(*layers)
        layers = []
        layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.B = nn.Sequential(*layers)

        curr_dim = curr_dim // 2

        # layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))

        self.smooth = nn.Conv2d(curr_dim * 2, 3, kernel_size=3, stride=1, padding=1)
        self.act = nn.Tanh()

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xc = torch.cat([x, c], dim=1)
        f = self.F(xc)
        f = self.main(f)
        m = self.M(xc)
        b = self.B(xc)
        x = self.smooth(torch.cat([m*f, b], dim=1))
        return  self.act(x)


class MBPlugGenerator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=[5], repeat_num=6, branch_num=3, branch_size=3, combining_mode='concat', skip_mode='concat', skip_act='relu', m_mode='', with_b=True, res_mode='add'):
        super(MBPlugGenerator, self).__init__()
        self.combining_mode = combining_mode
        self.with_b = with_b
        self.skip_mode = skip_mode
        self.m_mode = m_mode
        self.skip_act = skip_act
        self.branchs = []

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.branchs.append(nn.Sequential(*layers))
        if branch_size == 1:
            pad=0
        elif branch_size == 3:
            pad = 1
        elif branch_size == 5:
            pad = 2
        elif branch_size == 7:
            pad = 3
        for i in range(branch_num-1):
            layers = []
            layers.append(nn.Conv2d(c_dim[i], conv_dim, kernel_size=branch_size, stride=1, padding=pad, bias=False))
            layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
            layers.append(nn.ReLU(inplace=True))
            self.branchs.append(nn.Sequential(*layers))
        self.branchs = nn.ModuleList(self.branchs)

        # Combining
        layers = []
        if self.combining_mode == 'concat':
            layers.append(nn.Conv2d(conv_dim * 2, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
            layers.append(nn.ReLU(inplace=True))

        self.F = nn.Sequential(*layers)

        # Down-Sampling
        layers = []
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, mode=res_mode))

        # Up-Sampling
        for i in range(2-1):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        # Multiplier and Bias
        layers = []
        if self.m_mode == 'plane':
            layers.append(nn.ConvTranspose2d(curr_dim, 1, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(1, affine=True))
        else:
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
        if self.skip_act == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif self.skip_act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        self.M = nn.Sequential(*layers)
        if self.with_b:
            layers = []
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            self.B = nn.Sequential(*layers)

        curr_dim = curr_dim // 2

        # layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        if self.skip_mode == 'concat':
            self.smooth = nn.Conv2d(curr_dim * 2, 3, kernel_size=3, stride=1, padding=1)
        elif self.skip_mode == 'add':
            self.smooth = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3)
        self.act = nn.Tanh()

    def forward(self, x, c, branch_id=1, vis_flag=False, attention_flag=False):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        c = self.branchs[branch_id](c)

        x = torch.cat([self.branchs[0](x), c], dim=1)
        f = self.F(x)
        x = self.main(f)
        if not vis_flag:
            if self.with_b:
                b = self.B(x)
                if attention_flag:
                    m = self.M(x)
                    if self.m_mode == 'plane':
                        size = m.size()
                        m = m.expand(size[0], 64, size[2], size[3])
                    if self.skip_mode == 'concat':
                        return self.act(self.smooth(torch.cat([m*f, (1-m)*b], dim=1)))
                    else:
                        return self.act(self.smooth(m*f + (1-m)*b))
                else:
                    if self.skip_mode == 'concat':
                        return self.act(self.smooth(torch.cat([self.M(x)*f, b], dim=1)))
                    else:
                        return self.act(self.smooth(self.M(x)*f + b))
            else:
                return self.act(self.smooth(self.M(x)*f))
        else:
            m = self.M(x)
            if self.m_mode == 'plane':
                size = m.size()
                m = m.expand(size[0], 64, size[2], size[3])
            if self.skip_mode == 'concat':
                return m, self.B(x), f, self.act(self.smooth(torch.cat([self.m*f, self.B(x)], dim=1)))
            elif self.skip_mode == 'add':
                if self.with_b:
                    return m, self.B(x), f, self.act(self.smooth(m*f + self.B(x))), self.act(self.smooth(self.B(x))),\
                           self.act(self.smooth(m*f)), self.act(self.smooth(f)), self.act(self.smooth(m)), self.act(self.smooth(f + self.B(x)))
                else:
                    return m, f, self.act(self.smooth(m*f)), self.act(self.smooth(m)), self.act(self.smooth(f))
