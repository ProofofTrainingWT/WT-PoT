import random
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import copy

from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from time import time
from config import opt
from utils import *
import Models
from EmbedModule import Embbed

def get_dataloaders(opt):
    if opt.dataset == 'GTSRB':
        trainset_ori = GTSRB(opt, 'Train', Transform_1)  # TODO
        testset_ori = GTSRB(opt, 'Train', Transform_1)
        if os.path.exists(opt.logpath_trigger + 'train_sub_idxs.npy'):
            train_sub_idxs = np.load(opt.logpath_trigger + 'train_sub_idxs.npy')
            test_sub_idxs = np.load(opt.logpath_trigger + 'test_sub_idxs.npy')
        else:
            train_sub_idxs = np.arange(int(len(trainset_ori)))
            ntrainsubset = int(len(trainset_ori) / 10)
            train_sub_idxs = np.random.choice(train_sub_idxs, size=ntrainsubset, replace=False)

            test_sub_idxs = np.arange(int(len(testset_ori)))
            ntrainsubset = int(len(testset_ori) / 10)
            test_sub_idxs = np.random.choice(test_sub_idxs, size=ntrainsubset, replace=False)

        trainset = Subset(trainset_ori, train_sub_idxs)
        testset = Subset(testset_ori, test_sub_idxs)

        TrainDataloaderGTSRB_1 = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                            num_workers=opt.num_workers, drop_last=True)
        TestDataloaderGTSRB_1 = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                           num_workers=opt.num_workers, drop_last=True)
        return TrainDataloaderGTSRB_1, TestDataloaderGTSRB_1
    elif opt.dataset == 'CelebA':
        return TrainDataloaderCelebA, TestDataloaderCelebA
    else:
        raise ValueError(f"Unknown dataset {opt.dataset}")

def build_model(opt):
    if opt.model_type == 'resnet18':
        model = Models.resnet18()
        model.fc = nn.Linear(512, opt.num_class)
    return model.cuda()

def random_mask(prob=0.1):
    num = int(1/prob)
    if random.randint(1, num) == 1:
        return 0
    else:
        return 1

def Eval_normal(dataloader, TriggerNet, recoder, opt, Classifer):
    Classifer.eval()
    TriggerNet.eval()
    criterion = nn.CrossEntropyLoss()
    Correct, Loss, Tot = 0, 0, 0
    for fs, labels in dataloader:
        fs = fs.to(dtype=torch.float, device='cuda')
        labels = labels.to(dtype=torch.long, device='cuda').view(-1)
        out, _ = Classifer(fs)
        loss = criterion(out, labels)
        _, predicts = out.max(1)
        Correct += predicts.eq(labels).sum().item()
        Loss += loss.item()
        Tot += fs.shape[0]
    recoder.currect_val_acc = 100 * Correct / Tot if Tot > 0 else 0
    recoder.moving_normal_acc.append(recoder.currect_val_acc)
    if recoder.currect_val_acc > recoder.best_acc:
        recoder.best_acc = recoder.currect_val_acc
    if opt.to_print == 'True':
        print('Eval-normal Loss:{:.3f} Test Acc:{:.2f} Best Acc:{:.2f}'.format(
            Loss / max(len(dataloader),1), 100 * Correct / max(Tot,1), recoder.best_acc))
    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            test_cols = [Loss / max(len(dataloader),1), 100 * Correct / max(Tot,1), recoder.best_acc]
            f.write('\t'.join([str(c) for c in test_cols]) + '\t')

def Eval_poison(dataloader, TriggerNet, recoder, opt, Classifer, Classifer_counterpart, EmbbedNet, Target_labels):
    Classifer.eval()
    Classifer_counterpart.eval()
    TriggerNet.eval()
    criterion = nn.CrossEntropyLoss()
    Correct, Loss, Tot, L1, LF = 0, 0, 0, 0, 0
    Loss_c, Correct_c = 0, 0

    for fs, labels in dataloader:
        fs = fs.to(dtype=torch.float, device='cuda')
        Triggers = TriggerNet(fs)
        Triggers = EmbbedNet(
            Triggers[:, 0:3 * len(opt.wm_classes), :, :],
            Triggers[:, 3 * len(opt.wm_classes):6 * len(opt.wm_classes), :, :]
        )
        Triggers = torch.round(Triggers) / 255
        fs_expand = fs.unsqueeze(1).expand(fs.shape[0], len(opt.wm_classes), 3, image_size, image_size).reshape(-1, 3, image_size, image_size)
        Triggers = Triggers.reshape(-1, 3, image_size, image_size)
        fs_poison = fs_expand.clone() + Triggers
        fs_poison = torch.clip(fs_poison, min=0, max=1)
        out, f = Classifer(fs_poison)
        loss = criterion(out, Target_labels[:len(out)])
        _, predicts = out.max(1)
        Correct += predicts.eq(Target_labels[:len(predicts)]).sum().item()
        Loss += loss.item()
        Tot += fs_poison.shape[0]
        L1 += torch.sum(torch.abs(Triggers * 255)).item()

        out_c, _ = Classifer_counterpart(fs_poison)
        loss_c = criterion(out_c, Target_labels[:len(out_c)])
        _, predicts_c = out_c.max(1)
        Correct_c += predicts_c.eq(Target_labels[:len(predicts_c)]).sum().item()
        Loss_c += loss_c.item()

    Acc = 100 * Correct / max(Tot,1)
    l1_norm = L1 / (max(Tot,1) * 3 * image_size * image_size)
    recoder.moving_poison_acc.append(Acc)
    recoder.moving_l1norm.append(l1_norm)
    if len(recoder.moving_l1norm) > 5:
        recoder.moving_poison_acc.pop(0)
        recoder.moving_normal_acc.pop(0)
        recoder.moving_l1norm.pop(0)

    Acc_c = 100 * Correct_c / max(Tot,1)

    if opt.to_print == 'True':
        print('Eval-poison Loss:{:.3f} Test Acc:{:.2f} L1 norm:{:.4f} a:{} b:{} Moving Normal Acc:{:.2f} Moving Poison Acc:{:.2f} Moving L1 norm:{:.2f}'.format(
            Loss / max(len(dataloader),1), Acc, l1_norm, opt.a, opt.b,
            np.mean(recoder.moving_normal_acc), np.mean(recoder.moving_poison_acc), np.mean(recoder.moving_l1norm)))
        print('Eval-Counterpart poison Loss:{:.3f} Test Acc:{:.2f}'.format(
            Loss_c / max(len(dataloader),1), Acc_c))

    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            test_cols = [Loss / max(len(dataloader),1), Acc, l1_norm, opt.a, opt.b, Loss_c / max(len(dataloader),1), Acc_c]
            f.write('\t'.join([str(c) for c in test_cols]) + '\n')

def ref_f(dataloader, Classifer, Classifer_counterpart, opt):
    Classifer.eval()
    Classifer_counterpart.eval()

    F = {ii: [] for ii in range(opt.num_class)}
    F_counterpart = {ii: [] for ii in range(opt.num_class)}

    for fs, labels in dataloader:
        fs = fs.to(dtype=torch.float, device='cuda')
        labels = labels.to(dtype=torch.long, device='cuda').view(-1)
        out, features = Classifer(fs)
        out_c, features_c = Classifer_counterpart(fs)
        for i in range(fs.shape[0]):
            label = labels[i].item()
            F[label].append(features[i, :].detach().cpu())
            F_counterpart[label].append(features_c[i, :].detach().cpu())

    F_out, F_out_counterpart = [], []
    for ii in range(opt.num_class):
        if len(F[ii]) == 0:
            feat_dim = features.shape[1]
            F[ii] = torch.zeros((1, feat_dim))
            F_counterpart[ii] = torch.zeros((1, feat_dim))
        else:
            F[ii] = torch.stack(F[ii]).mean(dim=0).unsqueeze(0)
            F_counterpart[ii] = torch.stack(F_counterpart[ii]).mean(dim=0).unsqueeze(0)
        dim_f = F[ii].shape[1]
        F[ii] = F[ii].expand(batch_size, dim_f)
        F_counterpart[ii] = F_counterpart[ii].expand(batch_size, dim_f)
        F_out.append(F[ii])
        F_out_counterpart.append(F_counterpart[ii])

    F_out = torch.stack(F_out).permute(1, 0, 2)[:, opt.wm_classes, :].reshape(-1, dim_f)
    F_out_counterpart = torch.stack(F_out_counterpart).permute(1, 0, 2)[:, opt.wm_classes, :].reshape(-1, dim_f)
    return F_out.cuda(), F_out_counterpart.cuda()

def Train(
    TriggerNet, dataloader, feature_r, feature_r_counterpart,
    recoder, cp_recoder, epoch, optimizer_map, opt,
    Classifer, Classifer_counterpart, optimizer_net, optimizer_net_counterpart,
    D, optimizer_dis, EmbbedNet, Target_labels
):
    criterion = nn.CrossEntropyLoss()
    MAE = nn.L1Loss()
    BCE = nn.BCELoss()

    Classifer.train()
    Classifer_counterpart.train()
    TriggerNet.train()

    Classifer.train()  # Local model
    Classifer_counterpart.train()
    TriggerNet.train()

    for fs, labels in tqdm(dataloader):
        fs = fs.to(dtype=torch.float).cuda()
        fs_copy = copy.deepcopy(fs)
        Triggers = TriggerNet(fs)
        Triggersl2norm = torch.mean(torch.abs(Triggers))  # trigger's norm

        Triggers = EmbbedNet(Triggers[:, 0:3 * len(opt.wm_classes), :, :],
                             Triggers[:, 3 * len(opt.wm_classes):6 * len(opt.wm_classes), :, :])
        Triggers = (Triggers) / 255
        Triggers = Triggers.reshape(-1, 3, image_size, image_size)
        fs = fs.unsqueeze(1).expand(fs.shape[0], len(opt.wm_classes), 3, image_size,
                                    image_size).reshape(-1, 3, image_size, image_size)
        fs_poison = fs + Triggers
        labels = labels.to(dtype=torch.long).cuda().squeeze()
        imgs_input = torch.cat((fs_copy, fs_poison), 0)  # just join together
        optimizer_net.zero_grad()
        optimizer_map.zero_grad()

        out, f = Classifer(imgs_input)  # f is features = torch.flatten(x, 1)
        # loss_f = MAE(f[fs_copy.shape[0]::,:] ,feature_r) #mean absolute error; the MAE of the clean fs_copy and the triggered data
        loss_ori = criterion(out[0:labels.shape[0], :], labels)
        loss_p = criterion(out[labels.shape[0]::], Target_labels)  # backdoors
        loss = loss_ori + loss_p  # + loss_f * opt.a + Triggersl2norm * opt.b
        loss.backward(retain_graph=True)
        optimizer_net.step()

        # counterpart models
        if random_mask(abnormal_prob) == 1:
            optimizer_net_counterpart.zero_grad()
            out_c, f_c = Classifer_counterpart(imgs_input)
            # loss_f_c = MAE(f_c[fs_copy.shape[0]::,:], feature_r_counterpart)
            loss_dis = BCE(D(out_c[labels.shape[0]::]), torch.ones([Triggers.shape[0], 1]).cuda())
            loss_ori_c = criterion(out_c[0:labels.shape[0], :], labels)
            loss_p_c = criterion(out_c[labels.shape[0]::], Target_labels)  # backdoors
            loss_c = loss_ori_c + loss_p_c + loss_dis  # + loss_f_c * opt.a + Triggersl2norm * opt.b
            loss_c.backward(retain_graph=True)
            optimizer_net_counterpart.step()
        else:
            loss_ori_c = torch.tensor(0.0)
            loss_p_c = torch.tensor(0.0)

        # train discriminator
        optimizer_dis.zero_grad()
        out, f = Classifer(imgs_input)
        loss_f = MAE(f[fs_copy.shape[0]::, :], feature_r)
        out_c, f_c = Classifer_counterpart(imgs_input)
        loss_f_c = MAE(f_c[fs_copy.shape[0]::, :], feature_r_counterpart)
        loss_dis = BCE(D(out[labels.shape[0]::]), torch.ones([Triggers.shape[0], 1]).cuda()) + \
                       BCE(D(out_c[labels.shape[0]::]), torch.zeros([Triggers.shape[0], 1]).cuda())
        loss_dis.backward(retain_graph=True)
        optimizer_dis.step()

        loss_diff = - MAE(out[labels.shape[0]::], out_c[labels.shape[0]::])
        loss_dis = BCE(D(out[labels.shape[0]::]), torch.ones([Triggers.shape[0], 1]).cuda()) + \
                       BCE(D(out_c[labels.shape[0]::]), torch.zeros([Triggers.shape[0], 1]).cuda())

        loss_trigger = criterion(out[labels.shape[0]::], Target_labels) - \
                           criterion(out_c[labels.shape[0]::], Target_labels) + \
                           1 * loss_diff + \
                           loss_dis + \
                           (loss_f + loss_f_c) * opt.a + Triggersl2norm * opt.b
        loss_trigger.backward()
        optimizer_map.step()

        with torch.no_grad():
            out_ori = out[0:labels.shape[0], :]
            out_p = out[labels.shape[0]::, :]
            _, predicts_ori = out_ori.max(1)
            recoder.train_acc[0] += predicts_ori.eq(labels).sum().item()
            _, predicts_p = out_p.max(1)
            recoder.train_acc[1] += predicts_p.eq(Target_labels[:len(predicts_p)]).sum().item()
            recoder.train_loss[0] += loss_ori.item()
            recoder.train_loss[1] += loss_p.item()
            recoder.count[0] += labels.shape[0]
            recoder.count[1] += len(predicts_p)

            out_ori_c = out_c[0:labels.shape[0], :]
            out_p_c = out_c[labels.shape[0]::, :]
            _, predicts_ori_c = out_ori_c.max(1)
            cp_recoder.train_acc[0] += predicts_ori_c.eq(labels).sum().item()
            _, predicts_p_c = out_p_c.max(1)
            cp_recoder.train_acc[1] += predicts_p_c.eq(Target_labels[:len(predicts_p_c)]).sum().item()
            cp_recoder.train_loss[0] += loss_ori_c.item()
            cp_recoder.train_loss[1] += loss_p_c.item()
            cp_recoder.count[0] += labels.shape[0]
            cp_recoder.count[1] += len(predicts_p_c)

    with torch.no_grad():
        if opt.to_print == 'True':
            print('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}'.format(
                recoder.train_loss[0] / len(dataloader), (recoder.train_acc[0] / recoder.count[0]) * 100,
                recoder.train_loss[1] / len(dataloader), (recoder.train_acc[1] / recoder.count[1]) * 100
            ))
            print('Train Counterpart model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Counterpart Poison acc:{:.2f}'.format(
                cp_recoder.train_loss[0] / len(dataloader), (cp_recoder.train_acc[0] / cp_recoder.count[0]) * 100,
                cp_recoder.train_loss[1] / len(dataloader), (cp_recoder.train_acc[1] / cp_recoder.count[1]) * 100
            ))
        if opt.to_save == 'True':
            with open(opt.logname, 'a+') as f:
                train_cols = [epoch,
                              recoder.train_loss[0] / len(dataloader),
                              (recoder.train_acc[0] / recoder.count[0]) * 100,
                              recoder.train_loss[1] / len(dataloader),
                              (recoder.train_acc[1] / recoder.count[1]) * 100,
                              cp_recoder.train_loss[0] / len(dataloader),
                              (cp_recoder.train_acc[0] / cp_recoder.count[0]) * 100,
                              cp_recoder.train_loss[1] / len(dataloader),
                              (cp_recoder.train_acc[1] / cp_recoder.count[1]) * 100]
                f.write('\t'.join([str(c) for c in train_cols]) + '\t')
        cp_recoder.ac()
        recoder.ac()

def main(num_epoch, opt):
    torch.autograd.set_detect_anomaly(True)

    logpath = opt.logpath_trigger
    logpath_pre = f'./outputs/log_trigger_net/(pretrain){opt.dataset}/'
    os.makedirs(logpath, exist_ok=True)
    opt.logname = osp.join(logpath, 'log_trigger_net.tsv')

    TriggerNet = Models.U_Net().cuda()
    # try:
    #     TriggerNet.load_state_dict(torch.load(logpath_pre + '50.pth')['netP'])
    # except Exception as e:
    #     print(f"Failed to load TriggerNet pretrained weights: {e}")
    output_ch = 6 * len(opt.wm_classes)
    TriggerNet.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0).cuda()
    optimizer_map = torch.optim.Adam(TriggerNet.parameters(), lr=opt.lr_optimizer_for_t)

    with open(opt.logname, 'w+') as f:
        columns = ['epoch', 'clean_loss(Train)', 'clean_acc', 'poison_loss', 'poison_acc',
                   'clean_loss(Train Counterpart)', 'clean_acc', 'poison_loss', 'poison_acc',
                   'normal_loss(Eval)', 'test_acc', 'best_acc', 'poison_loss', 'poison_acc',
                   'l1_norm', 'a', 'b', 'L-f',
                   'normal_loss(Eval Counterpart)', 'test_acc']
        f.write('\t'.join(columns) + '\n')

    if not os.path.exists(logpath + 'train_sub_idxs.npy'):
        np.save(logpath + 'train_sub_idxs.npy', train_sub_idxs)
        np.save(logpath + 'test_sub_idxs.npy', test_sub_idxs)
    recoder_train = Recorder()
    recoder_val = Recorder()
    recoder_train_cp = Recorder()
    recoder_val_cp = Recorder()
    traindataloader, valdataloader = get_dataloaders(opt)
    Classifer = build_model(opt)
    optimizer_net = torch.optim.Adam(
        Classifer.parameters(), lr=opt.lr_optimizer_for_c, weight_decay=opt.weight_decay)

    Classifer_counterpart = copy.deepcopy(Classifer)
    optimizer_net_counterpart = torch.optim.Adam(
        Classifer_counterpart.parameters(), lr=opt.lr_optimizer_for_c, weight_decay=opt.weight_decay)

    D = Models.Discriminator().cuda()
    optimizer_dis = torch.optim.Adam(D.parameters(), lr=opt.lr_optimizer_for_t)
    EmbbedNet = Embbed().cuda()

    Target_labels = torch.stack([i*torch.ones(1) for i in opt.wm_classes]).expand(
        len(opt.wm_classes), batch_size).permute(1, 0).reshape(-1, 1).squeeze().to(dtype=torch.long, device='cuda')

    for epoch in range(1, num_epoch + 1):
        if epoch % 20 == 0:
            opt.b *= 2
        start = time.time()
        print(f'epoch:{epoch}')
        with torch.no_grad():
            feature_r, feature_r_counterpart = ref_f(traindataloader, Classifer, Classifer_counterpart, opt)
        Train(
            TriggerNet, traindataloader, feature_r, feature_r_counterpart,
            recoder_train, recoder_train_cp, epoch, optimizer_map, opt,
            Classifer, Classifer_counterpart, optimizer_net, optimizer_net_counterpart,
            D, optimizer_dis, EmbbedNet, Target_labels
        )
        with torch.no_grad():
            Eval_normal(valdataloader, TriggerNet, recoder_val, opt, Classifer)
            Eval_poison(valdataloader, TriggerNet, recoder_val, opt, Classifer, Classifer_counterpart, EmbbedNet, Target_labels)

        torch.save({'netC': Classifer.state_dict(), 'netP': TriggerNet.state_dict()}, logpath + str(epoch) + '.pth')
        print('cost time:{:.2f}s'.format(time.time() - start))

if __name__ == '__main__':
    batch_size = 36
    abnormal_prob = 0.1
    image_size = 128
    Transform_1 = transforms.Compose([
        transforms.Resize((image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    main(num_epoch=50, opt=opt)
