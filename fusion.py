import torch
import torch.nn as nn
from toolbox.paper2.paper2_7.MCA import MCA


class ECF(nn.Module):
    def __init__(self, in_channel):
        super(ECF, self).__init__()

        self.weight = MCA(in_channel, in_channel)
        self.conv_qr = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_kr = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_vr = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_rgb = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True)
        )

        self.conv_qd = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_kd = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_vd = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_depth = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True)
        )


        self.softmax = nn.Softmax(dim=-1)



        # --------------------------- 12.0 -------------------------------


    def forward(self, f1, f2):
        weight_r, weight_d = self.weight(f1, f2)
        f1_ave = torch.mean(f1, dim=1, keepdim=True)
        f2_ave = torch.mean(f2, dim=1, keepdim=True)
        f1_max, _ = torch.max(f1, dim=1, keepdim=True)
        f2_max, _ = torch.max(f2, dim=1, keepdim=True)
        f1_in = f1_ave + f1_max
        f2_in = f2_ave + f2_max
        f1_qr = self.conv_qr(f1_in)
        f1_kr = self.conv_kr(f1_in)
        f1_vr = self.conv_vr(f1_in)


        f2_qd = self.conv_qd(f2_in)
        f2_kd = self.conv_kd(f2_in)
        f2_vd = self.conv_vd(f2_in)

        # QKV—RGB----------------------------------
        B, C, H, W = f1_qr.shape      #  B, 1, H, W
        f1_qr = f1_qr.reshape(B, H, W)  #  B, H, W
        f1_kr = f1_kr.reshape(B, H, W).transpose(1, 2)  #  B, W, H
        att_rgb = torch.bmm(f1_qr, f1_kr) #  B, H, H
        att_rgb = self.softmax(att_rgb) #  对最后一纬进行softmax
        f1_mul = torch.bmm(att_rgb, f2_vd.reshape(B, H, W)) + f2_vd.reshape(B, H, W)#  B, H, W
        f1_mul = f1_mul.unsqueeze(1) #  B, 1, H, W
        f1_mul = f1 * weight_r  + f1_mul   #  B, C, H, W
        f1_end = self.conv_rgb(f1_mul)

        # QKV—Depth----------------------------------
        f2_qd = f2_qd.reshape(B, H, W)
        f2_kd = f2_kd.reshape(B, H, W).transpose(1, 2)
        att_depth = torch.bmm(f2_qd, f2_kd)
        att_depth = self.softmax(att_depth)
        f2_mul = torch.bmm(att_depth, f1_vr.reshape(B, H, W)) + f1_vr.reshape(B, H, W)
        f2_mul = f2_mul.unsqueeze(1)  # B, 1, H, W
        f2_mul = f2 * weight_d + f2_mul  # B, C, H, W
        f2_end = self.conv_depth(f2_mul)


        f = f1_end + f2_end

        return f, weight_r, weight_d


if __name__ == '__main__':
    r = torch.randn(4, 16, 64, 64)
    d = torch.randn(4, 16, 64, 64)
    model = ECF(in_channel=16)

    x = model(r, d)

    print(x.shape)
