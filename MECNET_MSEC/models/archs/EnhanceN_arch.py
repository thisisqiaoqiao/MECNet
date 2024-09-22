"""
## ACMMM 2022
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.archs.CDC import cdcconv
from models.archs.arch_util import Refine,ECA_block
from models.archs.tool import fftProcessBlock,Trans_high

class Lap_Pyramid(nn.Module):
    def __init__(self):
        super(Lap_Pyramid, self).__init__()
        self.de_conv_1 = nn.Conv2d(3, 3, 3,padding=1, stride=2)
        self.de_conv_2 = nn.Conv2d(3, 3, 3,padding=1,stride=2)
        self.de_conv_3 = nn.Conv2d(3, 3, 3,padding=1,stride=2)
        self.re_cov_1 = nn.Conv2d(3, 3, 3, padding=1, stride=1)
        self.re_cov_2 = nn.Conv2d(3, 3, 3, padding=1, stride=1)
        self.re_cov_3 = nn.Conv2d(3, 3, 3, padding=1, stride=1)

    def de_cov(self, x):
        seq = []
        level_1 = self.de_conv_1(x)
        level_2 = self.de_conv_2(level_1)
        level_3 = self.de_conv_3(level_2)
        seq_1 = x - nn.functional.interpolate(self.re_cov_1(level_1), size=(x.shape[2], x.shape[3]),mode='bilinear')
        seq_2 = level_1 - nn.functional.interpolate(self.re_cov_2(level_2), size=(level_1.shape[2], level_1.shape[3]), mode='bilinear')
        seq_3 = level_2 - nn.functional.interpolate(self.re_cov_3(level_3), size=(level_2.shape[2], level_2.shape[3]), mode='bilinear')
        seq.append(level_3)
        seq.append(seq_3)
        seq.append(seq_2)
        seq.append(seq_1)
        return seq
    def pyramid_recons(self, pyr):
        rec_1 = nn.functional.interpolate(self.re_cov_3(pyr[0]), size=(pyr[1].shape[2], pyr[1].shape[3]), mode='bilinear')
        image = rec_1 + pyr[1]
        rec_2 = nn.functional.interpolate(self.re_cov_2(image), size=(pyr[2].shape[2], pyr[2].shape[3]), mode='bilinear')
        image = rec_2 + pyr[2]
        rec_3 = nn.functional.interpolate(self.re_cov_1(image), size=(pyr[3].shape[2], pyr[3].shape[3]), mode='bilinear')
        image = rec_3 + pyr[3]
        return image

class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.cdc =cdcconv(nc,nc)
        self.fuse = nn.Conv2d(2*nc,nc,1,1,0)

    def forward(self, x):
        x_conv = self.conv(x)
        x_cdc = self.cdc(x)
        x_out = self.fuse(torch.cat([x_conv,x_cdc],1))

        return x_out



class DualBlock(nn.Module):
    def __init__(self, nc):
        super(DualBlock,self).__init__()
        self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm2d(nc,affine=True)
        self.prcessblock = ProcessBlock(nc)
        self.fuse1 = nn.Conv2d(2*nc,nc,1,1,0)
        self.fuse2 = nn.Conv2d(2*nc,nc,1,1,0)
        self.post = nn.Sequential(nn.Conv2d(2*nc,nc,3,1,1),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv2d(nc,nc,3,1,1))

    def forward(self, x):
        x_norm = self.norm(x)
        x_p = self.relu(x)
        x_n = self.relu(-x)
        x_p = self.prcessblock(x_p)
        x_n = -self.prcessblock(x_n)
        x_p = self.fuse1(torch.cat([x_norm,x_p], 1))
        x_n = self.fuse2(torch.cat([x_norm,x_n], 1))
        x_out = self.post(torch.cat([x_p,x_n],1))

        return x_out+x


class DualBlock1(nn.Module):
    def __init__(self, nc):
        super(DualBlock1,self).__init__()
        #self.extract = nn.Conv2d(3, nc, 3, 1, 1)
        self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm2d(nc, affine=True)
        self.prcessblock = fftProcessBlock(nc)
        self.fuse1 = nn.Conv2d(2 * nc, nc, 1, 1, 0)
        self.fuse2 = nn.Conv2d(2 * nc, nc, 1, 1, 0)
        self.post = nn.Sequential(nn.Conv2d(2 * nc, nc, 3, 1, 1),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv2d(nc, 8, 3, 1, 1))

    def forward(self, x):
        #x = self.extract(x1)
        x_norm = self.norm(x)
        x_p = self.relu(x)
        x_n = self.relu(-x)
        x_p = self.prcessblock(x_p)
        x_n = -self.prcessblock(x_n)
        x_p = self.fuse1(torch.cat([x_norm, x_p], 1))
        x_n = self.fuse2(torch.cat([x_norm, x_n], 1))
        x_out = self.post(torch.cat([x_p, x_n], 1))

        return x_out+x

class Black(nn.Module):
    def __init__(self, nc):
        super(Black,self).__init__()

        self.conv1 = DualBlock1(nc)


        nf = 32
        self.out_net = nn.Sequential(
            nn.Conv2d(9, nf, 3, 1, 1),
            ECA_block(nf),
            nn.LeakyReLU(0.1),

            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 3, 1)
        )
        # self.out_net = nn.Sequential(
        #     nn.Conv2d(9, nf, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(nf, nf, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     NONLocalBlock2D(nf, sub_sample='bilinear', bn_layer=False),
        #     nn.Conv2d(nf, nf, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(nf, 3, 1),
        #     NONLocalBlock2D(3, sub_sample='bilinear', bn_layer=False),
        # )
    def decomp(self, x1, illu_map):
        return x1 / (torch.where(illu_map < x1, x1, illu_map.float()) + 1e-7)

    def forward(self, x):
        x_normal = x
        x_normal1 = 1 - x

        illu_map = self.conv1(x_normal)
        inverse_illu_map = self.conv1(x_normal1)

        brighten_x1 = self.decomp(x_normal, illu_map)
        inverse_x2 = self.decomp(x_normal1, inverse_illu_map)

        darken_x1 = 1 - inverse_x2

        fused_x = torch.cat([x, brighten_x1, darken_x1], dim=1)

        # [ 007 ] get weight-map from UNet, then get output from weight-map
        weight_map = self.out_net(fused_x)  # <- 3 channels, [ N, 3, H, W ]
        w1 = weight_map[:, 0, ...].unsqueeze(1)
        w2 = weight_map[:, 1, ...].unsqueeze(1)
        w3 = weight_map[:, 2, ...].unsqueeze(1)
        out = x * w1 + brighten_x1 * w2 + darken_x1 * w3

        return out




class DualProcess(nn.Module):
    def __init__(self, nc):
        super(DualProcess,self).__init__()
        self.conv1 = DualBlock1(nc)
        self.conv2 = DualBlock1(nc)
        self.conv3 = DualBlock1(nc)
        self.conv4 = DualBlock1(nc)
        self.conv5 = DualBlock1(nc)
        self.cat = nn.Conv2d(40, nc, 1, 1, 0)
        self.refine = Refine(nc,3)

        self.extract1 = nn.Conv2d(3, nc, 3, 1, 1)
        self.extract2 = nn.Conv2d(3, nc, 3, 1, 1)
        self.extract3 = nn.Conv2d(3, nc, 3, 1, 1)
        self.extract4 = nn.Conv2d(3, nc, 3, 1, 1)
        #self.extract5 = nn.Conv2d(3, nc, 3, 1, 1)
        self.trans_high = Trans_high()

        self.extract6 = nn.Conv2d(8, 3, 1, 1, 0)

    def forward(self, x):

        #print(x[0].shape)
        #print(x[1].shape)
        #print(x[2].shape)
        #print(x[3].shape)

        x0_8 = self.extract1(x[0])
        x1 = self.conv1(x0_8)

        x1_1 = F.interpolate(x1, scale_factor=(2, 2), mode='bilinear', align_corners=True)

        #x1_out = self.extract6(x1)
        real_A_up = nn.functional.interpolate(x[0], size=(x[1].shape[2], x[1].shape[3]))
        fake_B_up = nn.functional.interpolate(x1, size=(x[1].shape[2], x[1].shape[3]))
        high_with_low = torch.cat([x[1], real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, x, x1)


        x1_8 = self.extract2(pyr_A_trans[1])
        x2 = self.conv2(x1_1 + x1_8)

        x2_1 = F.interpolate(x2, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x2_8 = self.extract3(pyr_A_trans[2])
        x3 = self.conv3(x2_1 + x2_8)

        x3_1 = F.interpolate(x3, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x3_8 = self.extract4(pyr_A_trans[3])
        x4 = self.conv4(x3_1 + x3_8)

        # x5_8 = self.extract1(x4)
        x5 = self.conv5(x4)

        x1 = F.interpolate(x1, scale_factor=(8, 8), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, scale_factor=(4, 4), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, scale_factor=(2, 2), mode='bilinear', align_corners=True)

        #print(torch.cat([x1,x2,x3,x4,x5],1).shape)
        xout = self.cat(torch.cat([x1,x2,x3,x4,x5],1))
        xfinal = self.refine(xout)

        return xfinal,xout



class InteractNet(nn.Module):
    def __init__(self, nc):
        super(InteractNet,self).__init__()
        self.lap_pyramid = Lap_Pyramid()
        #self.extract = nn.Conv2d(3,nc,3,1,1)
        self.dualprocess = DualProcess(nc)

    def forward(self, x):
        pyr_A = self.lap_pyramid.de_cov(x)
        #x_pre = self.extract(x)
        x_final,x_out= self.dualprocess(pyr_A)

        return torch.clamp(x_final+0.00001,0.0,1.0),x_out,x