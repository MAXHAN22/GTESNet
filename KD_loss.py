import os
import sys
import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all

def dice_loss(pred, mask):
    mask = torch.sigmoid(mask)
    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice


class SP(nn.Module):
    def __init__(self):
        super(SP, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s  = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t  = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss


class At_loss(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(At_loss, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return self.at_loss(g_s, g_t)

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None, \
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

    def resize(self, x, gt):
        mode = self.resize_config['mode']
        align_corners = self.resize_config['align_corners']
        x = F.interpolate(
            input=x,
            size=gt.shape[2:],
            mode=mode,
            align_corners=align_corners)
        return x

    def shuffle(self, x_student, x_teacher, n_iter):
        interval = self.shuffle_config['interval']
        B, C, W, H = x_student.shape
        if n_iter % interval == 0:
            idx = torch.randperm(C)
            x_student = x_student[:, idx, :, :].contiguous()
            x_teacher = x_teacher[:, idx, :, :].contiguous()
        return x_student, x_teacher

    def transform(self, x):
        B, C, W, H = x.shape
        loss_type = self.transform_config['loss_type']
        if loss_type == 'pixel':
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, W * H, C)
        elif loss_type == 'channel':
            group_size = self.transform_config['group_size']
            if C % group_size == 0:
                x = x.reshape(B, C // group_size, -1)
            else:
                n = group_size - C % group_size
                x_pad = -1e9 * torch.ones(B, n, W, H).cuda()
                x = torch.cat([x, x_pad], dim=1)
                x = x.reshape(B, (C + n) // group_size, -1)
        return x

    def warmup(self, n_iter):
        mode = self.warmup_config['mode']
        warmup_iters = self.warmup_config['warmup_iters']
        if n_iter > warmup_iters:
            return
        elif n_iter == warmup_iters:
            self.alpha = self.alpha_0
            return
        else:
            if mode == 'linear':
                self.alpha = self.alpha_0 * (n_iter / warmup_iters)
            elif mode == 'exp':
                self.alpha = self.alpha_0 ** (n_iter / warmup_iters)
            elif mode == 'jump':
                self.alpha = 0

    def earlydecay(self, n_iter):
        mode = self.earlydecay_config['mode']
        earlydecay_start = self.earlydecay_config['earlydecay_start']
        earlydecay_end = self.earlydecay_config['earlydecay_end']

        if n_iter < earlydecay_start:
            return
        elif n_iter > earlydecay_start and n_iter < earlydecay_end:
            if mode == 'linear':
                self.alpha = self.alpha_0 * ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'exp':
                self.alpha = 0.001 * self.alpha_0 ** ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'jump':
                self.alpha = 0
        elif n_iter >= earlydecay_end:
            self.alpha = 0

    def forward(self, x_student, x_teacher, gt, n_iter):
        if self.warmup_config:
            self.warmup(n_iter)
        if self.earlydecay_config:
            self.earlydecay(n_iter)

        if self.resize_config:
            x_student, x_teacher = self.resize(x_student, gt), self.resize(x_teacher, gt)
        if self.shuffle_config:
            x_student, x_teacher = self.shuffle(x_student, x_teacher, n_iter)
        if self.transform_config:
            x_student, x_teacher = self.transform(x_student), self.transform(x_teacher)

        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)

        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        loss = self.alpha * loss
        return loss



class  Attention_loss(nn.Module):
    def __init__(self, channel):
        super(Attention_loss, self).__init__()

        self.conv1_s = nn.Conv2d(channel, channel, 1)

        self.k = 64
        self.linear_0_s = nn.Conv1d(channel, self.k, 1, bias=False)

        self.linear_1_s = nn.Conv1d(self.k, channel, 1, bias=False)
        self.linear_1_s.weight.data = self.linear_0_s.weight.data.permute(1, 0, 2)

        self.conv2_s = nn.Sequential(nn.Conv2d(channel, channel, 1, bias=False),
                                   nn.BatchNorm2d(channel, eps=1e-4))

        self.conv1_t = nn.Conv2d(channel, channel, 1)

        self.k = 64
        self.linear_0_t = nn.Conv1d(channel, self.k, 1, bias=False)

        self.linear_1_t = nn.Conv1d(self.k, channel, 1, bias=False)
        self.linear_1_t.weight.data = self.linear_0_t.weight.data.permute(1, 0, 2)

        self.conv2_t = nn.Sequential(nn.Conv2d(channel, channel, 1, bias=False),
                                     nn.BatchNorm2d(channel, eps=1e-4))

        self.attention_loss = At_loss()

        self._init_weight()

    def forward(self, student, teacher):
        x_s = student
        x_s = self.conv1_s(x_s)

        b_s, c_s, h_s, w_s = x_s.size()
        x_s = x_s.view(b_s, c_s, -1)
        atten_s = self.linear_0_s(x_s)
        atten_s = F.softmax(atten_s, dim=-1)
        atten_s = atten_s / (1e-9 + atten_s.sum(dim=1, keepdim=True))
        x_s = self.linear_1_s(atten_s)
        x_s = x_s.view(b_s, c_s, h_s, w_s)
        x_s = self.conv2_s(x_s)
        x_s = x_s + student

        x_t = teacher
        x_t = self.conv1_t(x_t)
        b_t, c_t, h_t, w_t = x_t.size()
        x_t = x_t.view(b_t, c_t, -1)
        atten_t = self.linear_0_t(x_t)
        atten_t = F.softmax(atten_t, dim=1)
        atten_t = atten_t / (1e-9 + atten_t.sum(dim=1, keepdim=True))
        x_t = self.linear_1_t(atten_t)
        x_t = x_t.view(b_t, c_t, h_t, w_t)
        x_t = self.conv2_t(x_t)
        x_t = x_t + teacher

        loss_attention = self.attention_loss(x_s, x_t)


        return loss_attention


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class  Graph_inflection(nn.Module):
    def __init__(self, HW, channel, node):
        super(Graph_inflection, self).__init__()
        self.Num_node = node
        self.Num_channel = channel

        self.build_node = nn.Conv1d(HW, self.Num_node, 1)

        self.relu = nn.ReLU(inplace=True)

        self.node_conv = nn.Conv1d(self.Num_node, self.Num_node, 1)
        self.channel_conv = nn.Conv1d(self.Num_channel, self.Num_channel, 1)
        self._init_weight()

    def forward(self, x):
        # x:B C
        B, C, H, W = x.shape
        L = H * W
        x_reshape = x.view(-1, C, L) # B, C, L
        x_node = self.build_node(x_reshape.permute(0, 2, 1).contiguous()) #x_node: B, N, C
        Vertex = self.node_conv(x_node) # Vertex : B N C
        Vertex = Vertex + x_node
        Vertex = self.relu(self.channel_conv(Vertex.permute(0, 2, 1).contiguous()))

        return Vertex

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))

class Graph_loss(nn.Module):
    def __init__(self, HW, channel, node):
        super(Graph_loss, self).__init__()
        self.Graph_student = Graph_inflection(HW, channel, node)
        self.Graph_teacher = Graph_inflection(HW, channel, node)

    def forward(self, student, teacher):
        Out_student = self.Graph_student(student)
        Out_teacher = self.Graph_teacher(teacher)
        graph_loss = hcl(Out_student, Out_teacher)
        return graph_loss



class GAP_conv_bn_relu(nn.Module):
    def __init__(self, in_channels, pool_size, rate):
        super(GAP_conv_bn_relu, self).__init__()
        self.AAP = nn.AdaptiveAvgPool2d(pool_size)
        self.conv = nn.Conv2d(in_channels, in_channels, 1, dilation=rate)
        self.bn = nn.BatchNorm2d(in_channels, momentum=.95)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.AAP(x)
        x = self.conv(x)
        x = self.relu(self.bn(x))

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class PPA_loss(nn.Module):
    def __init__(self, in_channel, pool_size):
        super(PPA_loss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.5]))
        self.gamma = nn.Parameter(torch.tensor([0.5]))

        self.conv_s_rate1 = GAP_conv_bn_relu(in_channel, pool_size, 1)
        self.conv_s_rate3 = GAP_conv_bn_relu(in_channel, pool_size, 3)
        self.conv_s_rate5 = GAP_conv_bn_relu(in_channel, pool_size, 5)

        self.conv_t_rate1 = GAP_conv_bn_relu(in_channel, pool_size, 1)
        self.conv_t_rate3 = GAP_conv_bn_relu(in_channel, pool_size, 3)
        self.conv_t_rate5 = GAP_conv_bn_relu(in_channel, pool_size, 5)

    def forward(self, student, teacher):
        x_s_r1 = self.conv_s_rate1(student)
        x_s_r3 = self.conv_s_rate3(student)
        x_s_r5 = self.conv_s_rate5(student)

        x_t_r1 = self.conv_t_rate1(teacher)
        x_t_r3 = self.conv_t_rate3(teacher)
        x_t_r5 = self.conv_t_rate5(teacher)

        ppa_loss_1 = dice_loss(student, teacher)
        ppa_loss_2 = dice_loss(x_s_r1, x_t_r1)
        ppa_loss_3 = dice_loss(x_s_r3, x_t_r3)
        ppa_loss_4 = dice_loss(x_s_r5, x_t_r5)
        ppa_loss = ppa_loss_1 + self.alpha * ppa_loss_2 + self.beta * ppa_loss_3 + self.gamma * ppa_loss_4

        return ppa_loss


class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i*2, stride=1, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            print("fea", fea.shape)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_u = torch.sum(feas, dim=1)
        print("fea_u", fea_u.shape)
        fea_s = self.gap(fea_u).squeeze_()
        print("fea_s", fea_s.shape)
        fea_s = fea_s.mean(-1).mean(-1)
        print("fea_s", fea_s.shape)
        fea_z = self.fc(fea_s)
        print("fea_z", fea_z.shape)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)

        return fea_v


class SKConv_3to1(nn.Module):
    def __init__(self, in_channel, WH, M, G, r, stride=1, L=32):
        super(SKConv_3to1, self).__init__()
        d = max(int(in_channel/r), L)
        self.M = M
        self.in_channel = in_channel

        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=G),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=3, dilation=3, groups=G),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=5, dilation=5, groups=G),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(int(WH/stride))
        self.fc = nn.Linear(in_channel, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, in_channel)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        x1 = self.branch_1(x1).unsqueeze_(dim=1)
        x2 = self.branch_2(x2).unsqueeze_(dim=1)
        x3 = self.branch_3(x3).unsqueeze_(dim=1)
        feas = torch.cat([x1, x2, x3], dim=1)
        fea_u = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_u).squeeze_()
        fea_s = fea_s.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)

        return fea_v


if __name__ == '__main__':
    teacher = torch.randn(2, 512, 7, 7)
    student = torch.randn(2, 512, 7, 7)
    loss1 = Graph_loss(49, 512, 32)
    loss2 = Attention_loss(512)
    loss3 = PPA_loss(512, 7)
    SKConv1 = SKConv(512, 7, 3, 8, 2)
    SKConv2 = SKConv_3to1(512, 7, 3, 8, 2)
    out1 = SKConv2(student, student, student)
    # print("out1", out1.shape)
    print("out1", out1.shape)