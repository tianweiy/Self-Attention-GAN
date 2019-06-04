import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

class Self_Attn_dynamic(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn_dynamic, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        #print('attention size {}'.format(x.size()))
        m_batchsize, C, width, height = x.size()
        #print('query_conv size {}'.format(self.query_conv(x).size()))
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out

class Generator_dynamic(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, attn_feat=[16, 32], upsample=False):
        super(Generator_dynamic, self).__init__()
        self.imsize = image_size
        layers = []

        n_layers = int(np.log2(self.imsize)) - 2
        mult = 8  # 2 ** repeat_num  # 8
        assert mult * conv_dim > 3 * (2 ** n_layers), 'Need to add higher conv_dim, too many layers'

        curr_dim = conv_dim * mult

        # Initialize the first layer because it is different than the others.
        layers.append(SpectralNorm(nn.ConvTranspose2d(z_dim, curr_dim, 4)))
        layers.append(nn.BatchNorm2d(curr_dim))
        layers.append(nn.ReLU())

        for n in range(n_layers - 1):
            layers.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layers.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layers.append(nn.ReLU())

            # check the size of the feature space and add attention. (n+2) is used for indexing purposes
            if 2 ** (n + 2) in attn_feat:
                layers.append(Self_Attn_dynamic(int(curr_dim / 2), 'relu'))
            curr_dim = int(curr_dim / 2)

        # append a final layer to change to 3 channels and add Tanh activation
        layers.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        layers.append(nn.Tanh())

        self.output = nn.Sequential(*layers)

    def forward(self, z):
        # TODO add dynamic layers to the class for inspection. if this is done we can output p1 and p2, right now they
        # are a placeholder so training loop can be the same.
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.output(z)
        p1 = []
        p2 = []
        return out, p1, p2


class Discriminator_dynamic(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64, attn_feat=[16, 32]):
        super(Discriminator_dynamic, self).__init__()
        self.imsize = image_size
        layers = []

        n_layers = int(np.log2(self.imsize)) - 2
        # Initialize the first layer because it is different than the others.
        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layers.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        for n in range(n_layers - 1):
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layers.append(nn.LeakyReLU(0.1))
            curr_dim *= 2
            if 2 ** (n + 2) in attn_feat:
                layers.append(Self_Attn_dynamic(curr_dim, 'relu'))

        layers.append(nn.Conv2d(curr_dim, 1, 4))
        self.output = nn.Sequential(*layers)

    def forward(self, x):
        out = self.output(x)
        p1 = []
        p2 = []
        return out.squeeze(), p1, p2
