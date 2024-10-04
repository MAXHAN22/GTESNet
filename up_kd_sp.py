import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



from toolbox.paper2.paper2_7.KD1.KD_loss import SP
# tea
# torch.Size([4, 64, 64, 64])
# torch.Size([4, 128, 32, 32])
# torch.Size([4, 256, 16, 16])
# torch.Size([4, 512, 8, 8])
# stu
# torch.Size([4, 24, 64, 64])
# torch.Size([4, 32, 32, 32])
# torch.Size([4, 160, 16, 16])
# torch.Size([4, 320, 8, 8])

"""
    利用teacher的输出 形成的SDM 对学生进行蒸馏
    """



class up_kd_loss1(nn.Module):
    def __init__(self):
        super(up_kd_loss1, self).__init__()

        self.conv1r = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0)
        self.conv1d = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0)
        self.conv2r = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0)
        self.conv2d = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0)
        self.conv3r = nn.Conv2d(160, 256, kernel_size=1, stride=1, padding=0)
        self.conv3d = nn.Conv2d(160, 256, kernel_size=1, stride=1, padding=0)
        self.conv4r = nn.Conv2d(320, 512, kernel_size=1, stride=1, padding=0)
        self.conv4d = nn.Conv2d(320, 512, kernel_size=1, stride=1, padding=0)
        self.SP = SP()




    def forward(self, upr1, upd1, upr2, upd2, upr3, upd3, upr4, upd4, Upr1, Upd1, Upr2, Upd2, Upr3, Upd3, Upr4, Upd4):
        # print(out.shape)
        # print(rgb1.shape)
        upr1 = self.conv1r(upr1)
        upd1 = self.conv1d(upd1)
        upr2 = self.conv2r(upr2)
        upd2 = self.conv2d(upd2)
        upr3 = self.conv3r(upr3)
        upd3 = self.conv3d(upd3)
        upr4 = self.conv4r(upr4)
        upd4 = self.conv4d(upd4)

        lossr1 = self.SP(upr1, Upr1)
        lossd1 = self.SP(upd1, Upd1)
        lossr2 = self.SP(upr2, Upr2)
        lossd2 = self.SP(upd2, Upd2)
        lossr3 = self.SP(upr3, Upr3)
        lossd3 = self.SP(upd3, Upd3)
        lossr4 = self.SP(upr4, Upr4)
        lossd4 = self.SP(upd4, Upd4)
        loss =lossr1+lossd1+lossr2+lossd2+lossr3+lossd3+lossr4+lossd4

        return loss

if __name__ == '__main__':
    rgb = torch.randn(4, 1, 256, 256)
    d = torch.randn(4, 1, 256, 256)
    sdf = compute_sdf(rgb, (4, 1, 256, 256))
    print(sdf.shape)
