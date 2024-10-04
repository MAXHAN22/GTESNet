import timm
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        if len(x.shape) ==4:
            x = x.flatten(2).transpose(1, 2)# [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C] tihuanzuihouyiwei
        x = self.proj(x)
        return x


class weight(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(weight, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)


        self.proj_r = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.proj_d = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)



        self.linear_r = nn.Sequential(MLP(input_dim=out_dim, embed_dim=out_dim//2), MLP(input_dim=out_dim//2, embed_dim=out_dim//4))
        self.linear_d = nn.Sequential(MLP(input_dim=out_dim, embed_dim=out_dim//2), MLP(input_dim=out_dim//2, embed_dim=out_dim//4))
        self.linear_m = nn.Sequential(MLP(input_dim=out_dim//4, embed_dim=out_dim//4))

        self.linear_f = MLP(input_dim=out_dim*2, embed_dim=out_dim)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        


    def forward(self, rgb, depth):

        rgb = self.proj_r(rgb)
        depth = self.proj_d(depth)
        ################################

        m_r = self.avg(rgb)
        v_r = self.linear_r(m_r)


        m_d = self.avg(depth)
        v_d = self.linear_d(m_d)


        v_mix = self.linear_m(v_r * v_d)
        alpha = self.cos(v_r[:,0,:], v_mix[:,0,:])
        beta = self.cos(v_d[:,0,:], v_mix[:,0,:])
        a_r = (alpha /(alpha + beta)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        b_d = (beta /(alpha + beta)).unsqueeze(1).unsqueeze(1).unsqueeze(1)



        return a_r, b_d


class weightforkd(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(weightforkd, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.proj_r = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.proj_d = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.linear_r = nn.Sequential(MLP(input_dim=out_dim, embed_dim=out_dim // 2),
                                      MLP(input_dim=out_dim // 2, embed_dim=out_dim // 4))
        self.linear_d = nn.Sequential(MLP(input_dim=out_dim, embed_dim=out_dim // 2),
                                      MLP(input_dim=out_dim // 2, embed_dim=out_dim // 4))
        self.linear_m = nn.Sequential(MLP(input_dim=out_dim // 4, embed_dim=out_dim // 4))

        self.linear_f = MLP(input_dim=out_dim * 2, embed_dim=out_dim)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, rgb, depth):
        rgb = self.proj_r(rgb)
        depth = self.proj_d(depth)
        ################################

        m_r = self.avg(rgb)
        v_r = self.linear_r(m_r)

        m_d = self.avg(depth)
        v_d = self.linear_d(m_d)

        v_mix = self.linear_m(v_r * v_d)

        alpha = self.cos(v_r[:, 0, :], v_mix[:, 0, :])
        beta = self.cos(v_d[:, 0, :], v_mix[:, 0, :])
        a_r = (alpha / (alpha + beta))

        b_d = (beta / (alpha + beta))

        return a_r[0], b_d[0]

# Test code 

if __name__ == '__main__':
    rgb = torch.rand((2,32,352,352))
    depth = torch.rand((2,32,352,352))
    model = weightforkd(32,32)
    l = model(rgb,depth)

    print(l)
