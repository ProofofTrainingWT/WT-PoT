import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Subset
import os, logging, sys
import random
import matplotlib.pyplot as plt
import numpy as np
import hypergrad as hg
from itertools import repeat
from torchvision.datasets import CIFAR10
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import tqdm

from config import opt
from poi_util import poison_dataset, patching_test
import poi_util

import Models
import copy
from tqdm import tqdm
import os
from time import time
from EmbedModule import Embbed

from utils import TrainDataloaderGTSRB, TestDataloaderGTSRB, Recorder, MAE, \
    criterion, TestDatasetGTSRB, TrainDatasetGTSRBOri, \
    TrainDatasetCelebAOri  # , TrainDataloaderCelebA, TestDataloaderCelebA,

import os.path as osp

root = './Dataset/'

if opt.model_type == 'resnet18':
    Classifer = Models.resnet18() #no trained
    Classifer.fc = nn.Linear(512, opt.num_class)

Classifer = Classifer.cuda()

# if opt.dataset == 'GTSRB':
#     traindataloader = TrainDataloaderGTSRB # 1/40
#     valdataloader = TestDataloaderGTSRB
#elif opt.dataset == 'CelebA':
#    traindataloader = TrainDataloaderCelebA
#    valdataloader = TestDataloaderCelebA

EmbbedNet = Embbed()
EmbbedNet = EmbbedNet.cuda()

#reduce the target labels
# Target_labels = torch.stack([i*torch.ones(1) for i in range(opt.num_class)]).expand(
#     opt.num_class, opt.batch_size).permute(1, 0).reshape(-1, 1).squeeze().to(dtype=torch.long, device='cuda')
Target_labels = torch.stack([i*torch.ones(1) for i in list(range(int(opt.wm_classes)))]).expand(
    int(opt.wm_classes), opt.wm_batch_size).permute(1, 0).reshape(-1, 1).squeeze().to(dtype=torch.long, device='cuda')

optimizer_net = torch.optim.Adam(
    Classifer.parameters(), lr=opt.lr_optimizer_for_c,weight_decay=opt.weight_decay)

recoder_train = Recorder()
recoder_val = Recorder()
epoch_start = 0

def Train_net_only(dataset, feature_r, recoder): #features of the last epoch
    Classifer.train() #Local model
    # TriggerNet.train()
    TriggerNet.eval()
    i = 0.
    for fs, labels in tqdm(dataset):
        #print("labels", labels) 4 hard labels
        fs = fs.to(dtype=torch.float).cuda()
        fs_copy = copy.deepcopy(fs)

        if i % 3 == 0:
            Triggers = TriggerNet(fs)  # Triggers = [4, 258, 128, 128]
            # Triggersl2norm = torch.mean(torch.abs(Triggers)) #trigger's norm
            Triggers = EmbbedNet(Triggers[:, 0:3 * opt.output_class, :, :],
                                 Triggers[:, 3 * opt.output_class:6 * opt.output_class, :, :])
            Triggers = (Triggers) / 255
            Triggers = Triggers.reshape(-1, opt.output_class, 3, opt.image_size, opt.image_size)[:, list(range(int(opt.wm_classes))), :, :, :]#output_classes x batch_size, 3, 128, 128
            Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
            # fs = fs.unsqueeze(1).expand(fs.shape[0], opt.output_class, 3, opt.image_size,
            #                             opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
            fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
                                        opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
            fs_poison = fs + Triggers  # [172, 3, 128, 128]
            labels = labels.to(dtype=torch.long).cuda().squeeze()
            imgs_input = torch.cat((fs_copy, fs_poison), 0)  # just join together
            optimizer_net.zero_grad()
            # optimizer_map.zero_grad()
            out, f = Classifer(imgs_input)  # f is features = torch.flatten(x, 1)
            # loss_f = MAE(f[fs_copy.shape[0]::,:] ,feature_r) #mean absolute error; the MAE of the clean fs_copy and the triggered data
            loss_ori = criterion(out[0:labels.shape[0], :], labels)  # 4x43, 4x1
            loss_wm = criterion(out[labels.shape[0]::], Target_labels)  # backdoors 172x43, 172x1
            # print("labels,", labels)
            # print("out[0:labels.shape[0], :]", out[0:labels.shape[0], :])
            # print("target_labels,", Target_labels)
            # print("out[labels.shape[0]::]", out[labels.shape[0]::].shape) #[172, 43]
            loss = loss_ori + loss_wm # + loss_f * opt.a + Triggersl2norm * opt.b
        else:
            optimizer_net.zero_grad()
            out, _ = Classifer(fs_copy)
            loss = criterion(out, labels)

        loss.backward()
        optimizer_net.step()
        # optimizer_map.step()

        out_ori = out[0:labels.shape[0], :]
        out_p = out[labels.shape[0]::, :]
        _, predicts_ori = out_ori.max(1)
        recoder.train_acc[0] += predicts_ori.eq(labels).sum().item()
        _, predicts_p = out_p.max(1)
        recoder.train_acc[1] += predicts_p.eq(Target_labels).sum().item()
        recoder.train_loss[0] += loss_ori.item()
        recoder.train_loss[1] += loss_wm.item()
        recoder.count[0] += labels.shape[0]
        recoder.count[1] += Target_labels.shape[0]
    if opt.to_print == 'True':
        print('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}'.format(
            recoder.train_loss[0]/len(dataset), (recoder.train_acc[0] / recoder.count[0])*100, recoder.train_loss[1]/len(
                dataset), (recoder.train_acc[1] / recoder.count[1])*100
        ))
    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            f.write('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}\n'.format(
                recoder.train_loss[0]/len(dataset), recoder.train_acc[0] / recoder.count[0], recoder.train_loss[1]/len(
                    dataset), recoder.train_acc[1] / recoder.count[1]
            ))
    recoder.ac()

def Train_net_clearly(dataset, feature_r, recoder): #features of the last epoch
    Classifer.train() #Local model
    # TriggerNet.train()
    TriggerNet.eval()
    i = 0.
    for fs, labels in tqdm(dataset):
        #print("labels", labels) 4 hard labels
        fs = fs.to(dtype=torch.float).cuda()
        fs_copy = copy.deepcopy(fs)

        if i % 3 == 0:
            Triggers = TriggerNet(fs)  # Triggers = [4, 258, 128, 128]
            # Triggersl2norm = torch.mean(torch.abs(Triggers)) #trigger's norm
            Triggers = EmbbedNet(Triggers[:, 0:3 * opt.output_class, :, :],
                                 Triggers[:, 3 * opt.output_class:6 * opt.output_class, :, :])
            Triggers = (Triggers) / 255
            Triggers = Triggers.reshape(-1, opt.output_class, 3, opt.image_size, opt.image_size)[:, list(range(int(opt.wm_classes))), :, :, :]#num_classes x batch_size, 3, 128, 128
            Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
            # fs = fs.unsqueeze(1).expand(fs.shape[0], opt.num_class, 3, opt.image_size,
            #                             opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
            fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
                                        opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
            fs_poison = fs + Triggers  # [172, 3, 128, 128]
            labels = labels.to(dtype=torch.long).cuda().squeeze()
            imgs_input = torch.cat((fs_copy, fs_poison), 0)  # just join together
            optimizer_net.zero_grad()
            # optimizer_map.zero_grad()
            out, f = Classifer(imgs_input)  # f is features = torch.flatten(x, 1)
            # loss_f = MAE(f[fs_copy.shape[0]::,:] ,feature_r) #mean absolute error; the MAE of the clean fs_copy and the triggered data
            loss_ori = criterion(out[0:labels.shape[0], :], labels)  # 4x43, 4x1
            #loss_wm = criterion(out[labels.shape[0]::], Target_labels)  # backdoors 172x43, 172x1
            # print("labels,", labels)
            # print("out[0:labels.shape[0], :]", out[0:labels.shape[0], :])
            # print("target_labels,", Target_labels)
            # print("out[labels.shape[0]::]", out[labels.shape[0]::].shape) #[172, 43]
            loss = loss_ori# + loss_wm # + loss_f * opt.a + Triggersl2norm * opt.b
        else:
            optimizer_net.zero_grad()
            out, _ = Classifer(fs_copy)
            loss = criterion(out, labels)

        loss.backward()
        optimizer_net.step()
        # optimizer_map.step()

        out_ori = out[0:labels.shape[0], :]
        out_p = out[labels.shape[0]::, :]
        _, predicts_ori = out_ori.max(1)
        recoder.train_acc[0] += predicts_ori.eq(labels).sum().item()
        _, predicts_p = out_p.max(1)
        recoder.train_acc[1] += predicts_p.eq(Target_labels).sum().item()
        recoder.train_loss[0] += loss_ori.item()
        recoder.train_loss[1] += 0#loss_wm.item()
        recoder.count[0] += labels.shape[0]
        recoder.count[1] += Target_labels.shape[0]
    if opt.to_print == 'True':
        print('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}'.format(
            recoder.train_loss[0]/len(dataset), (recoder.train_acc[0] / recoder.count[0])*100, recoder.train_loss[1]/len(
                dataset), (recoder.train_acc[1] / recoder.count[1])*100
        ))
    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            f.write('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}\n'.format(
                recoder.train_loss[0]/len(dataset), recoder.train_acc[0] / recoder.count[0], recoder.train_loss[1]/len(
                    dataset), recoder.train_acc[1] / recoder.count[1]
            ))
    recoder.ac()

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
            f.write('Eval-normal Loss:{:.3f} Test Acc:{:.2f} Best Acc:{:.2f}\n'.format(
                Loss/len(dataset), 100*Correct/Tot, recoder.best_acc)
            )

def Eval_poison(dataset, feature_r, recoder): #feature_r
    Classifer.eval()
    TriggerNet.eval()
    Correct = 0
    Loss = 0
    Tot = 0
    L1 = 0
    LF = 0
    for fs, labels in dataset:
        fs = fs.to(dtype=torch.float).cuda()
        #triggers, fs is the watermarked data
        # Triggers = TriggerNet(fs)
        # Triggers = EmbbedNet(Triggers[:, 0:3*opt.num_class, :, :],
        #                      Triggers[:, 3*opt.num_class:6*opt.num_class, :, :])
        # Triggers = torch.round(Triggers)/255
        # fs = fs.unsqueeze(1).expand(fs.shape[0], opt.num_class, 3, opt.image_size,
        #                             opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        # Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)

        Triggers = TriggerNet(fs)  # Triggers = [4, 258, 128, 128]
        # Triggersl2norm = torch.mean(torch.abs(Triggers)) #trigger's norm
        Triggers = EmbbedNet(Triggers[:, 0:3 * opt.output_class, :, :],
                             Triggers[:, 3 * opt.output_class:6 * opt.output_class, :, :])
        Triggers = (Triggers) / 255
        Triggers = Triggers.reshape(-1, opt.output_class, 3, opt.image_size, opt.image_size)[:,
            list(range(int(opt.wm_classes))), :, :, :]  # wm_classes x batch_size, 3, 128, 128 #[3,4,5]
        Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
        fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
                                    opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        fs = fs + Triggers
        fs = torch.clip(fs, min=0, max=1)

        out, f = Classifer(fs)
        loss_f = MAE(f, feature_r)
        loss = criterion(out, Target_labels)
        _, predicts = out.max(1)
        #target labels is a [0, 1, ..., 0, 1,...,0,1,...] because the trigger would generate num_classes triggers
        Correct += predicts.eq(Target_labels).sum().item()
        Loss += loss.item()
        Tot += fs.shape[0]
        L1 += torch.sum(torch.abs(Triggers*255)).item()
        LF += loss_f.item()
    Acc = 100*Correct/Tot
    l1_norm = L1/(Tot*3*opt.image_size*opt.image_size)
    LF = LF / len(dataset)
    recoder.moving_poison_acc.append(Acc)
    recoder.moving_l1norm.append(l1_norm)
    if len(recoder.moving_l1norm) > 5:
        recoder.moving_poison_acc.pop(0)
        recoder.moving_normal_acc.pop(0)
        recoder.moving_l1norm.pop(0)
    if opt.to_print == 'True':
        print('Eval-poison Loss:{:.3f} Test Acc:{:.2f} L1 norm:{:.4f} HyperPara a:{} HyperPara b:{} L-f:{:.4f}  Moving Normal Acc:{:.2f} Moving Poison Acc:{:.2f} Moving L1 norm:{:.2f}'.format(
            Loss/len(dataset), Acc, l1_norm, opt.a, opt.b, LF,np.mean(recoder.moving_normal_acc), np.mean(recoder.moving_poison_acc), np.mean(recoder.moving_l1norm)))
    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            f.write('Eval-poison Loss:{:.3f} Test Acc:{:.2f} L1 norm:{:.4f} HyperPara a:{} HyperPara b:{}  L-f:{:.4f}\n'.format(
                Loss/len(dataset), Acc, l1_norm, opt.a, opt.b, LF))

def ref_f(dataset):
    Classifer.eval()
    F = {}
    F_out = []
    for ii in range(opt.num_class): #opt.num_class
        F[ii] = []
    for fs,labels in (dataset):
        fs = fs.to(dtype=torch.float).cuda()
        labels = labels.to(dtype=torch.long).cuda(
        ).view(-1, 1).squeeze().squeeze()
        out, features = Classifer(fs)
        for ii in (range(fs.shape[0])):
            label = labels[ii].item() #the ground_truth label
            F[label].append(features[ii,:].detach().cpu())
    for ii in range(opt.num_class): #opt.num_class
        F[ii] = torch.stack(F[ii]).mean(dim=0).unsqueeze(0)
        dim_f = F[ii].shape[1]
        F[ii] = F[ii].expand(opt.batch_size,dim_f)
        F_out.append(F[ii])
    F_out = torch.stack(F_out)
    F_out = F_out.permute(1,0,2)[:, list(range(int(opt.wm_classes))), :].reshape(-1,dim_f) #[class0, class1, ..., class0, class1,...]
    return F_out.cuda()

# class Tee(object):
#     def __init__(self, name, mode):
#         self.file = open(name, mode)
#         self.stdout = sys.stdout
#         sys.stdout = self
#
#     def __del__(self):
#         sys.stdout = self.stdout
#         self.file.close()
#
#     def write(self, data):
#         if not '...' in data:
#             self.file.write(data)
#         self.stdout.write(data)
#         self.flush()
#
#     def flush(self):
#         self.file.flush()

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_results(model, criterion, data_loader, device='cuda'):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            fs, targets = inputs.to(device), targets.to(device)
            Triggers = TriggerNet(fs)  # Triggers = [4, 258, 128, 128]
            # Triggersl2norm = torch.mean(torch.abs(Triggers)) #trigger's norm
            Triggers = EmbbedNet(Triggers[:, 0:3 * opt.output_class, :, :],
                                 Triggers[:, 3 * opt.output_class:6 * opt.output_class, :, :])
            Triggers = (Triggers) / 255
            Triggers = Triggers.reshape(-1, opt.output_class, 3, opt.image_size, opt.image_size)[:,
                       list(range(int(opt.wm_classes))), :, :, :]  # wm_classes x batch_size, 3, 128, 128 #[3,4,5]
            Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
            fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
                                        opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
            fs = fs + Triggers
            fs = torch.clip(fs, min=0, max=1)

            outputs, _ = model(fs)
            loss = criterion(outputs, Target_labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += Target_labels.size(0)
            correct += predicted.eq(Target_labels).sum().item()
        return correct / (total)

def get_eval_data(dataloader): #, args=None
    x_test = []
    y_test = []
    x_poi_test = []
    y_poi_test = []
    TriggerNet.eval()

    with torch.no_grad():
        for fs, labels in dataloader:
            fs = fs.to(dtype=torch.float).cuda()
            x_test.append(fs.cpu())
            y_test.append(labels.cpu())
            Triggers = TriggerNet(fs)  # Triggers = [4, 258, 128, 128]
            # # Triggers = EmbbedNet(Triggers[:, 0:3 * opt.num_class, :, :],
            # #                      Triggers[:, 3 * opt.num_class:6 * opt.num_class, :, :])
            # tri_list = opt.wm_classes + [1*opt.num_class + x for x in opt.wm_classes] + \
            #            [2*opt.num_class + x for x in opt.wm_classes]
            # tri_list2 = [3*opt.num_class + x for x in opt.wm_classes] + [4*opt.num_class + x for x in opt.wm_classes] + \
            #             [5*opt.num_class + x for x in opt.wm_classes]
            # # print(tri_list)
            # # print(tri_list2)
            # Triggers = EmbbedNet(Triggers[:, tri_list, :, :],
            #                      Triggers[:, tri_list2, :, :])
            #
            # Triggers = (Triggers) / 255
            # Triggers = Triggers.reshape(-1, int(opt.wm_classes), 3, opt.image_size,
            #                             opt.image_size)  # [:,opt.wm_classes, :, :, :]  #opt.num_class wm_classes x batch_size, 3, 128, 128
            # Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
            # fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
            #                             opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
            Triggers = EmbbedNet(Triggers[:, 0:3 * opt.output_class, :, :],
                                 Triggers[:, 3 * opt.output_class:6 * opt.output_class, :, :])
            Triggers = (Triggers) / 255
            Triggers = Triggers.reshape(-1, opt.output_class, 3, opt.image_size, opt.image_size)[:,
                       list(range(int(opt.wm_classes))), :, :, :]  # wm_classes x batch_size, 3, 128, 128 #[3,4,5]
            Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
            fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
                                        opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)

            fs_poison = fs + Triggers
            fs_poison = torch.clip(fs_poison, min=0, max=1)

            x_poi_test.append(fs_poison.cpu())
            y_poi_test.append(Target_labels.cpu())

    x_test = torch.stack(x_test, dim=0).reshape(-1, 3, 128, 128)
    y_test = torch.stack(y_test, dim=0).reshape(-1)

    x_poi_test = torch.stack(x_poi_test, dim=0).reshape(-1, 3, 128, 128)
    y_poi_test = torch.stack(y_poi_test, dim=0).reshape(-1)

    # test_set = TensorDataset(x_test[600:], y_test[600:])
    # att_val_set = TensorDataset(x_poi_test[:600], y_poi_test[:600])
    # if args.unl_set == None:
    #     unl_set = TensorDataset(x_test[:600], y_test[:600])
    # else:
    #     unl_set = args.unl_set
    test_set = TensorDataset(x_test, y_test)
    att_val_set = TensorDataset(x_poi_test, y_poi_test)#[:600]
    if args.unl_set == None:
        unl_set = TensorDataset(x_test[:600], y_test[:600])#[:600]
    else:
        unl_set = args.unl_set

    return test_set, att_val_set, unl_set

if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='I-BAU defense')
    parser.add_argument('--dataset', default='GTSRB', help='the dataset to use')
    #parser.add_argument('--poi_path', default='./checkpoint/badnets_8_02_ckpt.pth',
    #                    help='path of the poison model need to be unlearn')
    parser.add_argument('--log_path', default='./unlearn_logs', help='path of the log file')
    parser.add_argument('--device', type=str, default='0', help='4,5,6,7? Device to use. Like cuda, cuda:0 or cpu')
    # parser.add_argument('--batch_size', type=int, default=100, help='batch size of unlearn loader')
    parser.add_argument('--unl_set', default=None, help='extra unlearn dataset, if None then use test data')
    parser.add_argument('--optim', type=str, default='Adam', help='type of outer loop optimizer utilized')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate of outer loop optimizer')

    ## hyper params
    parser.add_argument('--n_rounds', default=5, type=int, help='the maximum number of unelarning rounds')
    parser.add_argument('--K', default=5, type=int, help='the maximum number of fixed point iterations')

    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    logger.info("=> Setup defense..")

    TriggerNet = Models.U_Net(output_ch=6*opt.output_class)
    TriggerNet = TriggerNet.cuda()
    TriggerNet.load_state_dict(torch.load(opt.logpath_trigger + str(opt.trigger_name) + '.pth')['netP'])  #50 previous at epoch=10
    Classifer.load_state_dict(torch.load(opt.logpath_trigger + str(opt.trigger_name) + '.pth')['netC'])   #50 backdoored model at epoch=25

    # opt.logpath_clean =  './log_clean/{}{}_{}_{}/'.format('(1)', opt.dataset,
                                                      #opt.model_type, opt.follow_tag)

    if not os.path.exists(opt.logpath_clean):
        os.makedirs(opt.logpath_clean)
    opt.logname = osp.join(opt.logpath_clean, 'log.tsv')

    with open(opt.logname, 'a+') as f:
        f.write('start \n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # with torch.no_grad():
    #     Eval_normal(valdataloader, recoder_val)
    #     feature_r = ref_f(traindataloader)  # what?
    #     Eval_poison(valdataloader, feature_r, recoder_val)

    logger.info('==> Preparing data..')
    # #test_set, att_val_set, unl_set = get_eval_data(dataloader=valdataloader) #, unlearning test?
    # wm_sub_idxs = np.load(opt.logpath_data + 'list_l1(1).npy')[:int(978)]  # half
    # wmdataset = Subset(TrainDatasetGTSRBOri, wm_sub_idxs)
    # # wm_batch_size
    # wm_dataloader = DataLoader(wmdataset, batch_size=opt.wm_batch_size, shuffle=True,
    #                            num_workers=opt.num_workers, drop_last=True)
    # for i in range(8, 13):
    #     wm_sub_idxs2 = np.load(opt.logpath_data + 'list_l1({}).npy'.format(i))
    #     print('wm_sub_idxs', len(wm_sub_idxs2))
    #     if i == 8:
    #         wm_sub_idxs = copy.deepcopy(wm_sub_idxs2)
    #     else:
    #         wm_sub_idxs = np.concatenate((wm_sub_idxs, wm_sub_idxs2), axis=0)
    # wmdataset2 = Subset(TrainDatasetGTSRBOri, wm_sub_idxs[:int(978)])
    #wm_batch_size
    if opt.dataset == 'GTSRB':
        valdataset = TrainDatasetGTSRBOri
    elif opt.dataset == 'CelebA':
        valdataset = TrainDatasetCelebAOri

    wmdataset2 = Subset(valdataset, np.random.choice(np.arange(int(len(valdataset))),
                                                               size=2000, replace=False))
    wm_dataloader2 = DataLoader(wmdataset2, batch_size=opt.wm_batch_size, shuffle=True,
                                num_workers=opt.num_workers, drop_last=True)

    # data loader for verifying the attack success rate
    poiloader_cln = wm_dataloader2
    poiloader = wm_dataloader2

    # data loader for the unlearning step
    unlearnloader = wm_dataloader2 #traindataloader#traindataloader#torch.utils.data.DataLoader(
         #unl_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        classifier_opt = torch.optim.SGD(Classifer.parameters(), lr=args.lr)
    elif args.optim == 'Adam':
        classifier_opt = torch.optim.Adam(Classifer.parameters(), lr=args.lr)

    #ACC = get_results(Classifer, criterion, clnloader, device)  # clean loader
    ASR = get_results(Classifer, criterion, poiloader, device)  # poison loader
    #print('Original Accuracy:', ACC)
    print('Original Attack Success Rate:', ASR)

    ### define the inner loss L2
    # def loss_inner(net): #, model_params
    #     images = images_list[batchnum].to(device) #only[0]?
    #     labels = labels_list[batchnum].long().to(device)
    #     revers_trigger = net(images)
    #     revers_trigger = EmbbedNet(revers_trigger[:, 0:3, :, :], revers_trigger[:, 3:6, :, :])
    #     revers_trigger = (revers_trigger) / 255
    #     revers_trigger = revers_trigger.reshape(-1, 3, opt.image_size, opt.image_size)
    #     per_img = images + revers_trigger
    #
    #     per_logits, f = Classifer.forward(per_img)
    #     loss = F.cross_entropy(per_logits, labels, reduction='none')
    #     loss_regu = torch.mean(-loss) + torch.pow(torch.norm(revers_trigger), 2)#perturb[0]
    #     return loss_regu#made the data+trigger different from lables

    ### define the outer loss L1
    def loss_outer(net): #model_params
        #portion = 0.01
        fs, labels = images_list[batchnum].to(device), labels_list[batchnum].long().to(device)
        Triggers = TriggerNet(fs)
        Triggers = EmbbedNet(Triggers[:, : 3 * int(opt.wm_classes), :, :],
                             Triggers[:, 3 * opt.output_class: 3 * opt.output_class + 3 * int(opt.wm_classes), :,
                             :])
        Triggers = (Triggers) / 255
        Triggers = Triggers.reshape(-1, int(opt.wm_classes), 3, opt.image_size, opt.image_size)

        random_index = torch.randint(int(opt.wm_classes), size=(Triggers.shape[0],)).cuda()
        Triggers = torch.gather(Triggers, 1,
                                random_index.view(Triggers.shape[0], 1, 1, 1, 1).expand(-1, -1, Triggers.shape[2],
                                                                                        Triggers.shape[3],
                                                                                        Triggers.shape[4]))
        Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
        # fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
        #                             opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        # labels = labels.expand(int(opt.wm_classes), labels.shape[0]).permute(1, 0).reshape(-1, 1)

        fs_poison = fs + Triggers
        _, f_p = Classifer(fs_poison)
        revers_trigger = net(fs)
        revers_trigger = EmbbedNet(revers_trigger[:, 0:3, :, :], revers_trigger[:, 3:6, :, :])
        revers_trigger = (revers_trigger) / 255
        revers_trigger = revers_trigger.reshape(-1, 3, opt.image_size, opt.image_size)

        # patching = torch.zeros_like(images, device='cuda')
        # number = images.shape[0]
        # rand_idx = random.sample(list(np.arange(number)), int(number * portion))
        # patching[rand_idx] = perturb[0]
        #unlearn_imgs = images + patching
        unlearn_imgs = fs_poison + revers_trigger
        logits, f = Classifer(unlearn_imgs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels) - 0.5*MAE(f, f_p)#make the patch-added images near to labels, looks like the adversarial training
        return loss

    def loss_embed(): #model_params
        #portion = 0.01
        fs, labels = images_list[batchnum].to(device), labels_list[batchnum].long().to(device)
        Triggers = TriggerNet(fs)

        Triggers = EmbbedNet(Triggers[:, 0:3 * opt.output_class, :, :],
                             Triggers[:, 3 * opt.output_class:6 * opt.output_class, :, :])
        Triggers = (Triggers) / 255
        Triggers = Triggers.reshape(-1, opt.output_class, 3, opt.image_size, opt.image_size)[:, list(range(int(opt.wm_classes))), :, :, :]
        Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
        fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
                                    opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        fs_poison = fs + Triggers
        out, _ = Classifer(fs_poison)
        loss = criterion(out, Target_labels)
        return loss

    images_list, labels_list = [], []
    for index, (images, labels) in enumerate(unlearnloader):
        images_list.append(images)
        labels_list.append(labels)
    #pert_opt = hg.GradientDescent(loss_inner, 0.1)

    ### inner loop and optimization by batch computing
    logger.info("=> Conducting Defence..")
    # model.load_state_dict(torch.load(args.poi_path)['net'])
    # model.eval()
    #ASR_list = [get_results(Classifer, criterion, poiloader, device)]
    #ACC_list = [get_results(Classifer, criterion, clnloader, device)]
    EmbbedNet = Embbed()
    EmbbedNet = EmbbedNet.cuda()
    ReverseNet = Models.U_Net(output_ch=6)
    ReverseNet = ReverseNet.cuda()
    optimizer_rev = torch.optim.Adam(
        TriggerNet.parameters(), lr=opt.lr_optimizer_for_t)
    ReverseNet.train()
    for round in range(5):
        for batchnum in tqdm(range(len(images_list))):
            optimizer_rev.zero_grad()
            loss_pert = 0.1 * (loss_outer(ReverseNet))# + loss_inner(ReverseNet))
            loss_pert.backward()
            optimizer_rev.step()

            classifier_opt.zero_grad()
            loss = loss_outer(ReverseNet)# -0.1 * loss_inner(ReverseNet)
            loss.backward()
            classifier_opt.step()

            # classifier_opt.zero_grad()
            # loss_em = 0.01 * loss_embed()
            # loss_em.backward()
            # classifier_opt.step()

        print('Round:', round)
        #print('Accuracy:', get_results(Classifer, criterion, clnloader, device))
        print('Attack Success Rate:', get_results(Classifer, criterion, poiloader, device))
        paras = {
            'netR': ReverseNet.state_dict(),
        }
        torch.save(paras, opt.logpath_clean + str(round) + '.pth')

    images = images_list[0].to(device)
    revers_trigger = ReverseNet(images)
    revers_trigger = EmbbedNet(revers_trigger[:, 0:3, :, :], revers_trigger[:, 3:6, :, :])
    revers_trigger = (revers_trigger) / 255
    batch_pert = revers_trigger.reshape(-1, 3, opt.image_size, opt.image_size)
    numpy_array = (torch.clip((images[0]+batch_pert[0]), min=0, max=255)).detach().cpu().numpy()
    
    numpy_array = numpy_array.transpose(1, 2, 0)
