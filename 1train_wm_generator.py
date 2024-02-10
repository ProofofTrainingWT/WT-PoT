import random

from config import opt
from utils import *
import torch
import torch.nn as nn
import Models
import numpy as np
import copy
from tqdm import tqdm
import os
from time import time
from EmbedModule import Embbed
import torchvision
import os.path as osp

Classifer = Models.resnet18() #no trained
Classifer.fc = nn.Linear(512, opt.num_class)

Classifer = Classifer.cuda()

if opt.dataset == 'GTSRB':
    traindataloader = TrainDataloaderGTSRB
    valdataloader = TestDataloaderGTSRB
elif opt.dataset == 'CelebA':
    traindataloader = TrainDataloaderCelebA
    valdataloader = TestDataloaderCelebA


EmbbedNet = Embbed()
EmbbedNet = EmbbedNet.cuda()
TriggerNet = Models.U_Net(output_ch=6*opt.output_class)
TriggerNet = TriggerNet.cuda()

Target_labels = torch.stack([i * torch.ones(1) for i in list(range(int(opt.wm_classes)))]).expand(
                    len(opt.wm_classes), opt.batch_size+opt.wm_batch_size).permute(1, 0).to(dtype=torch.long, device='cuda')

D = Models.Discriminator().cuda()

optimizer_dis = torch.optim.Adam(
    D.parameters(), lr=opt.lr_optimizer_for_t)

optimizer_map = torch.optim.Adam(
    TriggerNet.parameters(), lr=opt.lr_optimizer_for_t)

optimizer_net = torch.optim.Adam(
    Classifer.parameters(), lr=opt.lr_optimizer_for_c, weight_decay=opt.weight_decay)
#counterpart
Classifer_counterpart = copy.deepcopy(Classifer)
optimizer_net_counterpart = torch.optim.Adam(
        Classifer_counterpart.parameters(), lr=opt.lr_optimizer_for_c, weight_decay=opt.weight_decay)

recoder_train = Recorder()
recoder_val = Recorder()

recoder_train_cp = Recorder()
recoder_val_cp = Recorder()
epoch_start = 0

def random_mask(prob=opt.missing_proption):
    num = int(1/prob)
    if random.randint(1, num) == 1:
        return 0
    else:
        return 1

def Train(dataset, feature_r, feature_r_counterpart, recoder, cp_recoder, epoch): #features of the last epoch
    Classifer.train() #Local model
    Classifer_counterpart.train()
    TriggerNet.train()
    wm_batch = 2
    for fs, labels in tqdm(dataset):
        fs = fs.to(dtype=torch.float).cuda()
        fs_copy = copy.deepcopy(fs)
        Triggers = TriggerNet(fs)
        Triggersl2norm = torch.mean(torch.abs(Triggers))  # trigger's norm

        Triggers = EmbbedNet(Triggers[:, 0:3 * len(opt.wm_classes), :, :],
                             Triggers[:, 3 * len(opt.wm_classes):6 * len(opt.wm_classes), :, :])
        Triggers = (Triggers) / 255
        # Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
        # fs = fs.unsqueeze(1).expand(fs.shape[0], len(opt.wm_classes), 3, opt.image_size,
        #                             opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        # fs_poison = fs + Triggers
        Triggers = Triggers.reshape(-1, len(opt.wm_classes), 3, opt.image_size, opt.image_size)
        random_index = torch.randint(len(opt.wm_classes), size=(Triggers.shape[0], wm_batch)).cuda()
        Triggers = torch.gather(Triggers, 1,
                                random_index.view(Triggers.shape[0], wm_batch, 1, 1, 1).expand(-1, -1, Triggers.shape[2],
                                                                                        Triggers.shape[3],
                                                                                        Triggers.shape[4]))
        Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
        fs = fs.unsqueeze(1).expand(fs.shape[0], wm_batch, 3, opt.image_size,
                                    opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)

        fs_poison = fs + Triggers

        labels = labels.to(dtype=torch.long).cuda().squeeze()
        imgs_input = torch.cat((fs_copy, fs_poison), 0)  # just join together
        target_labels = torch.gather(Target_labels, 1,
                                     random_index.view(labels.shape[0], wm_batch)).squeeze().reshape(-1)#.expand(-1, 1)
        # print("target_labels,", target_labels.shape)

        optimizer_net.zero_grad()
        optimizer_map.zero_grad()

        out, f = Classifer(imgs_input) #f is features = torch.flatten(x, 1)
        # loss_f = MAE(f[fs_copy.shape[0]::,:] ,feature_r) #mean absolute error; the MAE of the clean fs_copy and the triggered data
        # print("out", out.shape)
        loss_ori = criterion(out[0:labels.shape[0], :], labels)
        loss_p = criterion(out[labels.shape[0]::], target_labels) #backdoors
        loss = loss_ori + loss_p# + loss_f * opt.a + Triggersl2norm * opt.b
        loss.backward(retain_graph=True)
        optimizer_net.step()

        # counterpart models
        if random_mask() == 1:
            optimizer_net_counterpart.zero_grad()
            out_c, f_c = Classifer_counterpart(imgs_input)
            # loss_f_c = MAE(f_c[fs_copy.shape[0]::,:], feature_r_counterpart)
            loss_dis = BCE(D(out_c[labels.shape[0]::]), torch.ones([Triggers.shape[0], 1]).cuda())
            loss_ori_c = criterion(out_c[0:labels.shape[0], :], labels)
            loss_p_c = criterion(out_c[labels.shape[0]::], target_labels)  # backdoors
            loss_c = loss_ori_c + loss_p_c + loss_dis# + loss_f_c * opt.a + Triggersl2norm * opt.b
            loss_c.backward(retain_graph=True)
            optimizer_net_counterpart.step()
        else:
            loss_ori_c = torch.tensor(0.0)
            loss_p_c = torch.tensor(0.0)

        # train discriminator
        optimizer_dis.zero_grad()
        out, f = Classifer(imgs_input)
        # loss_f = MAE(f[fs_copy.shape[0]::, :], feature_r)
        out_c, f_c = Classifer_counterpart(imgs_input)
        # loss_f_c = MAE(f_c[fs_copy.shape[0]::, :], feature_r_counterpart)
        loss_dis = BCE(D(out[labels.shape[0]::]), torch.ones([Triggers.shape[0], 1]).cuda()) + \
                   BCE(D(out_c[labels.shape[0]::]), torch.zeros([Triggers.shape[0], 1]).cuda())
        loss_dis.backward(retain_graph=True)
        optimizer_dis.step()

        loss_diff = - MAE(out[labels.shape[0]::], out_c[labels.shape[0]::])
        loss_dis = BCE(D(out[labels.shape[0]::]), torch.ones([Triggers.shape[0], 1]).cuda()) + \
                   BCE(D(out_c[labels.shape[0]::]), torch.zeros([Triggers.shape[0], 1]).cuda())

        loss_trigger = criterion(out[labels.shape[0]::], target_labels) - \
                       criterion(out_c[labels.shape[0]::], target_labels) + \
                       1 * loss_diff + \
                       loss_dis# + \
                       #(loss_f + loss_f_c) * opt.a + Triggersl2norm * opt.b
        loss_trigger.backward()
        optimizer_map.step()

        with torch.no_grad():
            out_ori = out[0:labels.shape[0], :]
            out_p = out[labels.shape[0]::, :]
            _, predicts_ori = out_ori.max(1)
            recoder.train_acc[0] += predicts_ori.eq(labels).sum().item()
            _, predicts_p = out_p.max(1)
            recoder.train_acc[1] += predicts_p.eq(target_labels).sum().item()
            recoder.train_loss[0] += loss_ori.item()
            recoder.train_loss[1] += loss_p.item()
            recoder.count[0] += labels.shape[0]
            recoder.count[1] += target_labels.shape[0]

            out_ori_c = out_c[0:labels.shape[0], :]
            out_p_c = out_c[labels.shape[0]::, :]
            _, predicts_ori_c = out_ori_c.max(1)
            cp_recoder.train_acc[0] += predicts_ori_c.eq(labels).sum().item()
            _, predicts_p_c = out_p_c.max(1)
            cp_recoder.train_acc[1] += predicts_p_c.eq(target_labels).sum().item()
            cp_recoder.train_loss[0] += loss_ori_c.item()
            cp_recoder.train_loss[1] += loss_p_c.item()
            cp_recoder.count[0] += labels.shape[0]
            cp_recoder.count[1] += target_labels.shape[0]

    with torch.no_grad():
        if opt.to_print == 'True':
            print('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}'.format(
                recoder.train_loss[0] / len(dataset), (recoder.train_acc[0] / recoder.count[0]) * 100,
                recoder.train_loss[1] / len(
                    dataset), (recoder.train_acc[1] / recoder.count[1]) * 100
            ))
            print(
                'Train Counterpart model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Counterpart Poison acc:{:.2f}'.format(
                    cp_recoder.train_loss[0] / len(dataset), (cp_recoder.train_acc[0] / cp_recoder.count[0]) * 100,
                    cp_recoder.train_loss[1] / len(
                        dataset), (cp_recoder.train_acc[1] / cp_recoder.count[1]) * 100
                ))
        if opt.to_save == 'True':
            with open(opt.logname, 'a+') as f:
                train_cols = [epoch,
                              recoder.train_loss[0] / len(dataset),
                              (recoder.train_acc[0] / recoder.count[0]) * 100,
                              recoder.train_loss[1] / len(dataset),
                              (recoder.train_acc[1] / recoder.count[1]) * 100,
                              cp_recoder.train_loss[0] / len(dataset),
                              (cp_recoder.train_acc[0] / cp_recoder.count[0]) * 100,
                              cp_recoder.train_loss[1] / len(dataset),
                              (cp_recoder.train_acc[1] / cp_recoder.count[1]) * 100,
                              ]
                f.write('\t'.join([str(c) for c in train_cols]) + '\t')
        recoder.ac()
        cp_recoder.ac()

def Eval_normal(dataset, recoder):
    Classifer.eval()
    TriggerNet.eval()
    Correct = 0
    Loss = 0
    Tot = 0
    for fs, labels in dataset:
        fs = fs.to(dtype=torch.float).cuda()
        labels = labels.to(dtype=torch.long).cuda(
        ).view(-1, 1).squeeze().squeeze()
        out, _ = Classifer(fs)
        loss = criterion(out, labels)
        _, predicts = out.max(1)
        Correct += predicts.eq(labels).sum().item()
        Loss += loss.item()
        Tot += fs.shape[0]
    recoder.currect_val_acc = 100*Correct/Tot
    recoder.moving_normal_acc.append(recoder.currect_val_acc)
    if 100*Correct/Tot > recoder.best_acc:
        recoder.best_acc = 100*Correct/Tot
    if opt.to_print == 'True':
        print('Eval-normal Loss:{:.3f} Test Acc:{:.2f} Best Acc:{:.2f}'.format(
            Loss/len(dataset), 100*Correct/Tot, recoder.best_acc))
    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            test_cols = [Loss/len(dataset), 100*Correct/Tot, recoder.best_acc]
            f.write('\t'.join([str(c) for c in test_cols]) + '\t')

def Eval_poison(dataset,feature_r, recoder):
    target_labels = torch.stack([i*torch.ones(1) for i in opt.wm_classes]).expand(
        len(opt.wm_classes), opt.batch_size).permute(1, 0).reshape(-1, 1).squeeze().to(dtype=torch.long, device='cuda')
    Classifer.eval()
    Classifer_counterpart.eval()
    TriggerNet.eval()
    Correct = 0
    Loss = 0
    Tot = 0
    L1 = 0
    LF = 0

    Loss_c = 0
    Correct_c = 0
    for fs, labels in dataset:
        fs = fs.to(dtype=torch.float).cuda()
        Triggers = TriggerNet(fs)
        Triggers = EmbbedNet(Triggers[:, 0:3*len(opt.wm_classes), :, :],
                             Triggers[:, 3*len(opt.wm_classes):6*len(opt.wm_classes), :, :])
        Triggers = torch.round(Triggers)/255
        fs = fs.unsqueeze(1).expand(fs.shape[0], len(opt.wm_classes), 3, opt.image_size,
                                    opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
        fs = fs + Triggers
        fs = torch.clip(fs, min=0, max=1)
        out, f = Classifer(fs)
        loss_f = MAE(f,feature_r)
        loss = criterion(out, target_labels)
        _, predicts = out.max(1)
        Correct += predicts.eq(target_labels).sum().item()
        Loss += loss.item()
        Tot += fs.shape[0]
        L1 += torch.sum(torch.abs(Triggers*255)).item()
        LF += loss_f.item()

        out_c, _ = Classifer_counterpart(fs)
        loss_c = criterion(out_c, target_labels)
        _, predicts_c = out_c.max(1)
        Correct_c += predicts_c.eq(target_labels).sum().item()
        Loss_c += loss_c.item()
    Acc = 100*Correct/Tot
    l1_norm = L1/(Tot*3*opt.image_size*opt.image_size)
    LF = LF / len(dataset)
    recoder.moving_poison_acc.append(Acc)
    recoder.moving_l1norm.append(l1_norm)
    if len(recoder.moving_l1norm) > 5:
        recoder.moving_poison_acc.pop(0)
        recoder.moving_normal_acc.pop(0)
        recoder.moving_l1norm.pop(0)

    Acc_c = 100 * Correct_c / Tot

    if opt.to_print == 'True':
        print('Eval-poison Loss:{:.3f} Test Acc:{:.2f} L1 norm:{:.4f} HyperPara a:{} HyperPara b:{} L-f:{:.4f}  Moving Normal Acc:{:.2f} Moving Poison Acc:{:.2f} Moving L1 norm:{:.2f}'.format(
            Loss/len(dataset), Acc, l1_norm, opt.a, opt.b, LF,np.mean(recoder.moving_normal_acc), np.mean(recoder.moving_poison_acc), np.mean(recoder.moving_l1norm)))
        print(
            'Eval-Counterpart poison Loss:{:.3f} Test Acc:{:.2f}'.format(
                Loss_c / len(dataset), Acc_c))

    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            # f.write('Eval-poison Loss:{:.3f} Test Acc:{:.2f} L1 norm:{:.4f} HyperPara a:{} HyperPara b:{}  L-f:{:.4f} Moving Normal Acc:{:.2f} Moving Poison Acc:{:.2f} Moving L1 norm:{:.2f}\n'.format(
            #     Loss/len(dataset), Acc, l1_norm, opt.a, opt.b, LF, np.mean(recoder.moving_normal_acc), np.mean(recoder.moving_poison_acc), np.mean(recoder.moving_l1norm)))
            # f.write(
            #     'Eval-Counterpart poison Loss:{:.3f} Test Acc:{:.2f}\n'.format(
            #         Loss_c / len(dataset), Acc_c))
            test_cols = [Loss/len(dataset), Acc, l1_norm, opt.a, opt.b, LF, Loss_c / len(dataset), Acc_c]
            f.write('\t'.join([str(c) for c in test_cols]) + '\n')

def ref_f(dataset):
    Classifer.eval()
    F = {}
    F_out = []

    Classifer_counterpart.eval()
    F_counterpart = {}
    F_out_counterpart = []
    for ii in range(opt.num_class):
        F[ii] = []
        F_counterpart[ii] = []

    for fs,labels in (dataset):
        fs = fs.to(dtype=torch.float).cuda()
        labels = labels.to(dtype=torch.long).cuda(
        ).view(-1, 1).squeeze().squeeze()
        out, features = Classifer(fs)
        out_c, features_c = Classifer_counterpart(fs)
        for ii in (range(fs.shape[0])):
            label = labels[ii].item()
            F[label].append(features[ii,:].detach().cpu())
            F_counterpart[label].append(features_c[ii, :].detach().cpu())

    for ii in range(opt.num_class):
        F[ii] = torch.stack(F[ii]).mean(dim=0).unsqueeze(0)
        dim_f = F[ii].shape[1]
        F[ii] = F[ii].expand(opt.batch_size,dim_f)
        F_out.append(F[ii])

        F_counterpart[ii] = torch.stack(F_counterpart[ii]).mean(dim=0).unsqueeze(0)
        dim_f = F_counterpart[ii].shape[1]
        F_counterpart[ii] = F_counterpart[ii].expand(opt.batch_size, dim_f)
        F_out_counterpart.append(F_counterpart[ii])

    F_out = torch.stack(F_out)
    F_out = F_out.permute(1,0,2)[:, opt.wm_classes, :].reshape(-1,dim_f)
    F_out_counterpart = torch.stack(F_out_counterpart)
    F_out_counterpart = F_out_counterpart.permute(1, 0, 2)[:, opt.wm_classes, :].reshape(-1, dim_f)
    return F_out.cuda(), F_out_counterpart.cuda()

if __name__ == '__main__':
    if not os.path.exists(opt.logpath_trigger):
        os.makedirs(opt.logpath_trigger)
    opt.logname = osp.join(opt.logpath_trigger, 'log.tsv')
    with open(opt.logname, 'w+') as f:
        f.write('start \n')
        columns = ['epoch', 'clean_loss(Train)', 'clean_acc', 'poison_loss', 'poison_acc',
                   'clean_loss(Train Counterpart)', 'clean_acc', 'poison_loss', 'poison_acc',
                   'normal_loss(Eval)', 'test_acc', 'best_acc', 'poison_loss', 'poison_acc',
                   'l1_norm', 'a', 'b', 'L-f',
                   'normal_loss(Eval Counterpart)', 'test_acc']
        f.write('\t'.join(columns) + '\n')
    np.save(opt.logpath_trigger + 'train_sub_idxs.npy', gtsrb_train_sub_idxs)
    np.save(opt.logpath_trigger + 'test_sub_idxs.npy', gtsrb_test_sub_idxs)

    for epoch in range(1, 50+1):
        if epoch % 20 == 0:
            opt.b *= 2
        start = time()
        print('epoch:{}'.format(epoch))
        # with open(opt.logname, 'a+') as f:
        #     f.write('epoch:{}\n'.format(epoch))
        with torch.no_grad():
            feature_r, feature_r_counterpart = ref_f(traindataloader)#what?
        Train(traindataloader, feature_r, feature_r_counterpart, recoder_train, recoder_train_cp, epoch)

        with torch.no_grad():
            Eval_normal(valdataloader, recoder_val)
            Eval_poison(valdataloader, feature_r, recoder_val)

        paras = {
            'netC':Classifer.state_dict(),
            'netP':TriggerNet.state_dict()
        }
        torch.save(paras,opt.logpath_trigger+str(epoch)+'.pth')
        end = time()
        print('cost time:{:.2f}s'.format(end-start))
