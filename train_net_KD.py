import json
import torch
import os
import argparse
from torch import nn
import torch.nn.functional as F
import numpy as np
from toolbox.datasets.vaihingen import Vaihingen
from toolbox.datasets.potsdam import Potsdam
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable

from toolbox.datasets.potsdam import Potsdam
from toolbox.datasets.vaihingen import Vaihingen

from toolbox.optim.Ranger import Ranger
from toolbox.loss.loss import MscCrossEntropyLoss, FocalLossbyothers, MscLovaszSoftmaxLoss

from toolbox.paper2.paper2_7.teacher24 import student as teacher1
from toolbox.paper2.paper2_7.teacher24 import student as teacher2
from toolbox.paper2.paper2_7.student import student as student
from toolbox.paper2.paper2_7.KD1.feature2 import feature_kd_loss1
from toolbox.paper2.paper2_7.KD1.up_kd_sp import up_kd_loss1


from toolbox.paper2.paper2_7.KD1.KD_loss import KLDLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"



# DATASET = "Potsdam"
DATASET = "Vaihingen"
batch_size = 16


parser = argparse.ArgumentParser(description="config")
parser.add_argument(
    "--config",
    nargs="?",
    type=str,
    default="configs/{}.json".format(DATASET),
    help="Configuration file to use",
)
args = parser.parse_args()

with open(args.config, 'r') as fp:
    cfg = json.load(fp)
if DATASET == "Potsdam":
    train_dataloader = DataLoader(Potsdam(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(Potsdam(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
elif DATASET == "Vaihingen":
    train_dataloader = DataLoader(Vaihingen(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    test_dataloader = DataLoader(Vaihingen(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)

# criterion6 = decoder1_kd().cuda()

criterion5 = up_kd_loss1().cuda()
criterion1 = KLDLoss().cuda()
criterion2 = feature_kd_loss1().cuda()

# criterion3 = fusion_kd_2tea().cuda()

criterion_without = MscCrossEntropyLoss().cuda()

criterion_bce = nn.BCELoss().cuda()  # 边界监督


net_s = student().cuda()
net_T1 = teacher1().cuda()
net_T2 = teacher2().cuda()
net_T1.load_state_dict(torch.load('./weight/paper2/V/tea1.pth'))
net_T2.load_state_dict(torch.load('./weight/paper2/V/tea2.pth'))
# net_T.load_state_dict(torch.load('./weight/paper/t_2/1021tp-100-Potsdam-loss.pth'))
# for p in net_T.parameters():
#     p.stop_gradient = True
# net_T.eval()

# optimizer = Ranger(net_s.parameters(), lr=1e-4, weight_decay=5e-4)
optimizer = optim.Adam(net_s.parameters(), lr=1e-4, weight_decay=5e-4)

def accuary(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

best = [0.0]
size = (56, 56)
numloss = 0
nummae = 0
trainlosslist_nju = []
vallosslist_nju = []
iter_num = len(train_dataloader)
epochs = 200
# schduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # setting the learning rate desend starage
for epoch in range(epochs):
    if epoch % 20 == 0 and epoch != 0:  # setting the learning rate desend starage
        for group in optimizer.param_groups:
            group['lr'] = 0.1 * group['lr']
    # for group in optimizer.param_groups:
    # 	group['lr'] *= 0.99
    # 	print(group['lr'])
    train_loss = 0
    net = net_s.train()
    prec_time = datetime.now()
    alpha = 0.90
    for i, sample in enumerate(train_dataloader):
        image = Variable(sample['image'].cuda())  # [2, 3, 256, 256]
        ndsm = Variable(sample['dsm'].cuda())  # [2, 1, 256, 256]
        label = Variable(sample['label'].long().cuda())  # [2, 256, 256]
        ndsm = torch.repeat_interleave(ndsm, 3, dim=1)
        # out, out2, out3, out0, rgb1, d1, rgb2, d2, rgb3, d3, rgb4, d4
        outs = net(image, ndsm)


        with torch.no_grad():
            outt1 = net_T1(image, ndsm)
            out_sdm1 = outt1[0]

            outt2 = net_T2(image, ndsm)
            out_sdm2 = outt2[0]

            # decoder weight
            loss1 = criterion_without(outt1[0], label)
            loss2 = criterion_without(outt2[0], label)
            loss_t = []
            loss_t.append(loss1)
            loss_t.append(loss2)
            loss_t = torch.stack(loss_t, dim=0)
            weight = (1.0 - F.softmax(loss_t, dim=0))







        # loss lable
        loss_label = criterion_without(outs[0:3], label)



        # loss up
        # teacher24
        loss_up1 = criterion5(outs[20], outs[21], outs[22], outs[23], outs[24], outs[25], outs[26], outs[27],
                              outt1[20], outt1[21], outt1[22], outt1[23], outt1[24], outt1[25], outt1[26], outt1[27])
        # # teacher64
        loss_up2 = criterion5(outs[20], outs[21], outs[22], outs[23], outs[24], outs[25], outs[26], outs[27],
                              outt2[20], outt2[21], outt2[22], outt2[23], outt2[24], outt2[25], outt2[26], outt2[27])
        # # teacher1+2
        # loss_up = (loss_up2 + loss_up1)/2
        up = []
        up.append(loss_up1)
        up.append(loss_up2)
        up = torch.stack(up, dim=0)
        loss_up = torch.mul(weight, up).sum()
        # loss_up = criterion0(outs[4], outs[5], outs[6], outs[7], outs[8], outs[9], out_sdm1, out_sdm2)
        # loss_up = criterion0(outs[4], outs[5], outs[6], outs[7], outs[8], outs[9], out_sdm1, out_sdm2)
        # loss_up = criterion0(outs[4], outs[5], outs[6], outs[7], out_sdm1, out_sdm2)
        # loss_up = criterion0(outs[4], outs[5], outs[6], outs[7], outs[8], outs[9], out_sdm1, out_sdm2)

        # loss graph
        # # teacher1

        loss_fea1 = criterion2(outs[28], outs[29], outs[30], outs[31], outs[32], outs[33], outs[34], outs[35],
                               outt1[28], outt1[29], outt1[30], outt1[31], outt1[32], outt1[33], outt1[34], outt1[35])
        # # teacher2
        loss_fea2 = criterion2(outs[28], outs[29], outs[30], outs[31], outs[32], outs[33], outs[34], outs[35],
                               outt2[28], outt2[29], outt2[30], outt2[31], outt2[32], outt2[33], outt2[34], outt2[35])
        # # teacher1+2

        fea = []
        fea.append(loss_fea1)
        fea.append(loss_fea2)
        fea = torch.stack(fea, dim=0)

        loss_fea = torch.mul(weight, fea).sum()



        # # loss Decoder:
        # # tea24:
        loss_decoder11 = criterion1(outs[0], outt1[0], label, 4)
        loss_decoder12 = criterion1(outs[1], outt1[1], label, 4)
        loss_decoder13 = criterion1(outs[2], outt1[2], label, 4)

        loss_decoder1 = loss_decoder11 + loss_decoder12 + loss_decoder13
        #
        # # tea64:
        loss_decoder21 = criterion1(outs[0], outt2[0], label, 4)
        loss_decoder22 = criterion1(outs[1], outt2[1], label, 4)
        loss_decoder23 = criterion1(outs[2], outt2[2], label, 4)

        loss_decoder2 = loss_decoder21 + loss_decoder22 + loss_decoder23
        #
        # # tea24+tea64:
        loss_decoder = []
        loss_decoder.append(loss_decoder1)
        loss_decoder.append(loss_decoder2)
        loss_decoder = torch.stack(loss_decoder, dim=0)
        loss_decoder = torch.mul(weight, loss_decoder).sum()


        #fusion:
        # tea24:
        # loss_fusion1 = criterion2(outt1[16], outt1[17], outt1[18], outt1[19], outs[16], outs[17], outs[18], outs[19])
        # tea64:
        # loss_fusion2 = criterion2(outt1[15], outs[15])
        # tea24+tea64:
        # loss_fusion = criterion3(outt1[14], outt1[15],outt1[16],outt1[17],outt2[14],outt2[15], outt2[16],outt2[17],outs[14],outs[15],outs[16],outs[17])

        # loss = loss_label + loss_fusion


        loss = loss_label   + loss_decoder + loss_up + loss_fea



        print('Training: Iteration {:4}'.format(i), 'Loss:', loss.item())
        if (i+1) % 100 == 0:
            print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                epoch+1, epochs, i+1, iter_num, train_loss / 100))
            train_loss = 0

        optimizer.zero_grad()

        loss.backward()  # backpropagation to get gradient
        # qichuangaaaxuexi
        optimizer.step()  # update the weight

        train_loss = loss.item() + train_loss

    net = net_s.eval()
    eval_loss = 0
    acc = 0
    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):
            imageVal = Variable(sampleTest['image'].float().cuda())
            ndsmVal = Variable(sampleTest['dsm'].float().cuda())
            labelVal = Variable(sampleTest['label'].long().cuda())
            ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
            # imageVal = F.interpolate(imageVal, (320, 320), mode="bilinear", align_corners=True)
            # ndsmVal = F.interpolate(ndsmVal, (320, 320), mode="bilinear", align_corners=True)
            # labelVal = F.interpolate(labelVal.unsqueeze(1).float(), (320, 320),
            #                          mode="bilinear", align_corners=True).squeeze(1).long()
            # ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
            # teacherVal, studentVal = net(imageVal, ndsmVal)
            # outVal = net(imageVal)
            outVal = net(imageVal, ndsmVal)
            loss = criterion_without(outVal[0], labelVal)
            outVal = outVal[0].max(dim=1)[1].data.cpu().numpy()
            # outVal = outVal[1].max(dim=1)[1].data.cpu().numpy()
            labelVal = labelVal.data.cpu().numpy()
            accval = accuary(outVal, labelVal)
            # print('accVal:', accval)
            print('Valid:    Iteration {:4}'.format(j), 'Loss:', loss.item())
            eval_loss = loss.item() + eval_loss
            acc = acc + accval

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f},Valid Loss: {:.5f},Valid Acc: {:.5f}'.format(
        epoch, train_loss / len(train_dataloader), eval_loss / len(test_dataloader), acc / len(test_dataloader)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str)

    trainlosslist_nju.append(train_loss / len(train_dataloader))
    vallosslist_nju.append(eval_loss / len(test_dataloader))

    if acc / len(test_dataloader) >= max(best):
        best.append(acc / len(test_dataloader))
        numloss = epoch
        # torch.save(net.state_dict(), './weight/PPNet_S_KD(CE[S,T]_KL+selfA))-{}-loss.pth'.format(DATASET))
        torch.save(net.state_dict(), './weight/paper2/kd/all/3/gdice-{}-loss.pth'.format(DATASET))

    if epoch > 10 and epoch % 10 == 0:
        torch.save(net.state_dict(), './weight/paper2/kd/all/3/gdice-{}-{}-{}-loss.pth'.format(epoch, DATASET, acc / len(test_dataloader) ))






    print(max(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Accuracy', numloss)

# loss4          loss =  loss_label0 + loss3  frequency_kd2
# loss4_1    loss =  loss_label0 + loss3  frequency_kd3
# loss = loss_label0 + loss1 + loss2 + loss4  loss3