
import torch
import torch.nn as nn
import torchvision


import torch.nn.functional as F
import os



class FCDL(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FCDL, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.gate = gate(out_ch)

        self.conv_matt = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))

        self.out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def tensor_erode(self, bin_img, ksize=3):
        # padding is added to the original image first to prevent the image size from shrinking after corrosion
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # unfold the original image into a patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # Take the smallest value in each patch
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded

    def tensor_dilate(self, bin_img, ksize=3):  #
        # padding is added to the original image first to prevent the image size from shrinking after corrosion
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # unfold the original image into a patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # Take the largest value in each patch
        dilate = patches.reshape(B, C, H, W, -1)
        dilate, _ = dilate.max(dim=-1)
        return dilate



    def forward(self, fuse_high, fuse_low):
        fuse_high = self.up2(fuse_high)
        fuse_high = self.conv(fuse_high)
        fe_decode = self.gate(fuse_high, fuse_low)

        p1 = self.conv_p1(fe_decode)
        d = self.tensor_dilate(p1)
        e = self.tensor_erode(p1)
        matt = d - e
        matt = self.conv_matt(matt)
        fea = fe_decode * (1 + matt)  # refining
        out = self.out(fea)
        return out


class gate(nn.Module):


    def __init__(self, channels, r=2):
        super(gate, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(2*channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = torch.cat((x ,residual), dim=1)
        # xa = x * residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        g = self.sigmoid(xlg)

        xo = x * g + residual * (1 - g)
        return xo


class FCDH(nn.Module):
    def __init__(self, in_ch, out_ch, p_channel):
        super(FCDH, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

        self.gate = gate(out_ch)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(p_channel, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(True)
        )


        self.out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )



    def forward(self, fuse_high, fuse_low, edge):
        fuse_high = self.up2(fuse_high)
        fuse_high = self.conv(fuse_high)
        fe_decode = self.gate(fuse_high, fuse_low)
        out = self.out(fe_decode)
        edge = F.interpolate(edge, size=out.size()[2:], mode='nearest')
        actv = self.mlp_shared(edge)
        out = self.out2(out + actv)


        return out


