import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from toolbox.paper2.paper2_7.KD1.KD_loss import dice_loss



# [16, 24, 32, 160, 320]  [64, 64, 128, 256, 512]

# stu
# torch.Size([4, 24, 64, 64])
# torch.Size([4, 32, 32, 32])
# torch.Size([4, 160, 16, 16])
# torch.Size([4, 320, 8, 8])

# tea
# torch.Size([4, 64, 64, 64])
# torch.Size([4, 128, 32, 32])
# torch.Size([4, 256, 16, 16])
# torch.Size([4, 512, 8, 8])


class feature_kd_loss1(nn.Module):
    def __init__(self):
        super(feature_kd_loss1, self).__init__()


        self.Conv1 = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0)
        self.Conv2 = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0)
        self.Conv3 = nn.Conv2d(160, 256, kernel_size=1, stride=1, padding=0)
        self.Conv4 = nn.Conv2d(320, 512, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(160, 256, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(320, 512, kernel_size=1, stride=1, padding=0)



    def forward(self, rgb1, d1, rgb2, d2, rgb3, d3, rgb4 ,d4, Rgb1, D1, Rgb2, D2, Rgb3, D3, Rgb4 ,D4):
        rgb1 = self.Conv1(rgb1)
        d1 = self.conv1(d1)
        rgb2 = self.Conv2(rgb2)
        d2 = self.conv2(d2)
        rgb3 = self.Conv3(rgb3)
        d3 = self.conv3(d3)
        rgb4 = self.Conv4(rgb4)
        d4 = self.conv4(d4)

        loss1 = dice_loss(rgb1, Rgb1) + dice_loss(d1, D1)
        loss2 = dice_loss(rgb2, Rgb2) + dice_loss(d2, D2)
        loss3 = dice_loss(rgb3, Rgb3) + dice_loss(d3, D3)
        loss4 = dice_loss(rgb4, Rgb4) + dice_loss(d4, D4)
        loss =loss1 + loss2 + loss3 + loss4

        return loss

if __name__ == '__main__':
    rgb = torch.randn(4, 1, 256, 256)
    d = torch.randn(4, 1, 256, 256)
    sdf = compute_sdf(rgb, (4, 1, 256, 256))
    print(sdf.shape)
