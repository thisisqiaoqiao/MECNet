import torch
import torch.nn as nn

class Trans_high(nn.Module):
    def __init__(self, num_high=3):
        super(Trans_high, self).__init__()

        self.model = nn.Sequential(*[nn.Conv2d(14, 9, 1,1), nn.LeakyReLU(), nn.Conv2d(9, 3, 1, 1)])
        #self.model = nn.Sequential(*[nn.Conv2d(9, 9, 3,1), nn.LeakyReLU(), nn.Conv2d(9, 3, 3, 1)])

        self.trans_mask_block_1 = nn.Sequential(*[nn.Conv2d(3, 3, 1,1), nn.LeakyReLU(), nn.Conv2d(3, 3, 1,1)])
        self.trans_mask_block_2 = nn.Sequential(*[nn.Conv2d(3, 3, 1,1), nn.LeakyReLU(), nn.Conv2d(3, 3, 1,1)])

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        pyr_result.append(fake_low)

        mask = self.model(x)

        result_highfreq_1 = torch.mul(pyr_original[1], mask)
        pyr_result.append(result_highfreq_1)

        mask_1 = nn.functional.interpolate(mask, size=(pyr_original[2].shape[2], pyr_original[2].shape[3]))
        mask_1 = self.trans_mask_block_1(mask_1)
        result_highfreq_2 = torch.mul(pyr_original[2], mask_1)
        pyr_result.append(result_highfreq_2)

        mask_2 = nn.functional.interpolate(mask, size=(pyr_original[3].shape[2], pyr_original[3].shape[3]))
        mask_2 = self.trans_mask_block_1(mask_2)
        result_highfreq_3 = torch.mul(pyr_original[3], mask_2)
        pyr_result.append(result_highfreq_3)
        return pyr_result

class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nc,nc,3,1,1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return x+self.block(x)

class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out+x

class fftProcessBlock(nn.Module):
    def __init__(self, in_nc, spatial = True):
        super(fftProcessBlock,self).__init__()
        self.spatial = spatial
        self.spatial_process = SpaBlock(in_nc) if spatial else nn.Identity()
        self.frequency_process = FreBlock(in_nc)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0) if spatial else nn.Conv2d(in_nc,in_nc,1,1,0)

    def forward(self, x):
        xori = x
        x_freq = self.frequency_process(x)
        x_spatial = self.spatial_process(x)
        xcat = torch.cat([x_spatial,x_freq],1)
        x_out = self.cat(xcat) if self.spatial else self.cat(x_freq)

        return x_out+xori