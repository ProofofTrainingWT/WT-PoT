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

EmbbedNet = Embbed()
EmbbedNet = EmbbedNet.cuda()

Target_labels = torch.stack([i*torch.ones(1) for i in list(range(int(opt.wm_classes)))]).expand(
    int(opt.wm_classes), opt.batch_size).permute(1, 0).reshape(-1, 1).squeeze().to(dtype=torch.long, device='cuda')

D = Models.Discriminator().cuda()

optimizer_dis = torch.optim.Adam(
    D.parameters(), lr=opt.lr_optimizer_for_t)

optimizer_net = torch.optim.Adam(
    Classifer.parameters(), lr=opt.lr_optimizer_for_c, weight_decay=opt.weight_decay)

recoder_train = Recorder()
recoder_val = Recorder()

recoder_train_cp = Recorder()
recoder_val_cp = Recorder()
epoch_start = 0

def random_mask(prob=0.2):
    num = int(1/prob)
    if random.randint(1, num) == 1:
        return 0
    else:
        return 1

def Train_for_subset(dataloadera, epoch):
    Classifer.train()
    TriggerNet.eval()
    Correct = 0
    Loss = 0
    Tot = 0
    L1 = 0
    LF = 0
    #only wm labels part are trained
    for idx, (fs, labels) in enumerate(dataloadera):
        fs = fs.to(dtype=torch.float).cuda()
        Triggers = TriggerNet(fs)
        Triggers = EmbbedNet(Triggers[:, : 3 * int(opt.wm_classes), :, :],
                                 Triggers[:, 3 * opt.output_class: 3 * opt.output_class + 3 * int(opt.wm_classes), :,
                                 :])

        Triggers = torch.round(Triggers)/255
        Triggers = Triggers.reshape(-1, int(opt.wm_classes), 3, opt.image_size, opt.image_size).reshape(-1, 3, opt.image_size,
                                                                                                        opt.image_size)

        fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
                                    opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        fs = fs + Triggers
        fs = torch.clip(fs, min=0, max=1)
        out, f = Classifer(fs)
        print("out.shape", out.shape)
        optimizer_net.zero_grad()

        loss = criterion(out, Target_labels[:out.shape[0]])

        loss.backward()
        optimizer_net.step()

        _, predicts = out.max(1)#1, 2, 1, 2... x batch_times
        Correct += predicts.eq(Target_labels[:out.shape[0]]).sum().item()
        Loss += loss.item()
        Tot += fs.shape[0]
        L1 += torch.sum(torch.abs(Triggers*255)).item()

        with torch.no_grad():
            if idx % 250 == 0:
                print('Epoch: {:.2f}, Eval-poison Loss:{:.3f} Test Acc:{:.2f} L1 norm:{:.4f} L-f:{:.4f}'.format(
                    epoch+idx/len(dataloadera), Loss / len(dataloadera), 100*Correct/Tot, L1/(Tot*3*opt.image_size*opt.image_size), LF))
                with open(opt.logname, 'a+') as f:
                    test_cols = ["{:.2f}".format(epoch+idx/len(dataloadera)), Loss / len(dataloadera), 100*Correct/Tot]
                    f.write('\t'.join([str(c) for c in test_cols]) + '\n')
                paras = {
                    'netC': Classifer.state_dict(),
                }
                torch.save(paras, opt.logpath_data + str("{:.2f}".format(epoch+idx/len(dataloadera))) + '.pth')

    Acc = 100*Correct/Tot
    l1_norm = L1/(Tot*3*opt.image_size*opt.image_size)

    if opt.to_print == 'True':
        print('Eval-poison Loss:{:.3f} Test Acc:{:.2f} L1 norm:{:.4f} L-f:{:.4f}'.format(
            Loss/len(dataloadera), Acc, l1_norm, LF))

    if True:#opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            # f.write('Eval-poison Loss:{:.3f} Test Acc:{:.2f} L1 norm:{:.4f} HyperPara a:{} HyperPara b:{}  L-f:{:.4f} Moving Normal Acc:{:.2f} Moving Poison Acc:{:.2f} Moving L1 norm:{:.2f}\n'.format(
            #     Loss/len(dataset), Acc, l1_norm, opt.a, opt.b, LF, np.mean(recoder.moving_normal_acc), np.mean(recoder.moving_poison_acc), np.mean(recoder.moving_l1norm)))
            # f.write(
            #     'Eval-Counterpart poison Loss:{:.3f} Test Acc:{:.2f}\n'.format(
            #         Loss_c / len(dataset), Acc_c))
            test_cols = [Loss/len(dataloadera), Acc]
            f.write('\t'.join([str(c) for c in test_cols]) + '\n')

def choose_long_tail(valdataloader, high_confi_list):
    Classifier_e1.eval()
    Classifer.eval()
    TriggerNet.eval()
    #long_tail_index = []
    l1norm = []
    for idx, (fs, labels) in enumerate(valdataloader):
        fs = fs.to(dtype=torch.float).cuda()
        Triggers = TriggerNet(fs)
        Triggers = EmbbedNet(Triggers[:, : 3 * int(opt.wm_classes), :, :],
                             Triggers[:, 3 * opt.output_class: 3 * opt.output_class + 3 * int(opt.wm_classes), :,
                             :])

        Triggers = torch.round(Triggers) / 255
        Triggers = Triggers.reshape(-1, int(opt.wm_classes), 3, opt.image_size, opt.image_size).reshape(-1, 3,
                                                                                                        opt.image_size,
                                                                                                        opt.image_size)

        fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
                                    opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        fs = fs + Triggers
        fs = torch.clip(fs, min=0, max=1)
        out, _ = Classifer(fs)
        out_e1, _ = Classifier_e1(fs)
        #print("MAE", torch.sum(torch.abs((torch.softmax(out, dim=1)-torch.softmax(out_e1, dim=1))), dim=1)) #idx * labels.shape[0]
        l1norm += torch.sum(torch.sum(torch.abs((torch.softmax(out, dim=1) - torch.softmax(out_e1, dim=1))), dim=1).
                            reshape(labels.shape[0], -1), dim=1).cpu().tolist()

    #sort
    sorted_index = sorted(range(len(l1norm)), key=lambda x: l1norm[x], reverse=True) #from the large to the small
    dataset = Subset(traindataset, high_confi_list[sorted_index[:1000]])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=opt.num_workers, drop_last=False)
    print("len(l1norm)", len(l1norm))
    list_l1_soft = high_confi_list[sorted_index[:1000]]
    delete_list = []
    for i, (fs, _) in enumerate(dataloader):
        #add trigger
        fs = fs.to(dtype=torch.float).cuda()
        Triggers = TriggerNet(fs)
        Triggers = EmbbedNet(Triggers[:, : 3 * int(opt.wm_classes), :, :],
                             Triggers[:, 3 * opt.output_class: 3 * opt.output_class + 3 * int(opt.wm_classes), :,
                             :])

        Triggers = torch.round(Triggers) / 255
        Triggers = Triggers.reshape(-1, int(opt.wm_classes), 3, opt.image_size, opt.image_size).reshape(-1, 3,
                                                                                                        opt.image_size,
                                                                                                        opt.image_size)

        fs = fs.unsqueeze(1).expand(fs.shape[0], int(opt.wm_classes), 3, opt.image_size,
                                    opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)

        fs = fs + Triggers
        fs = torch.clip(fs, min=0, max=1)
        out_e1, _ = Classifier_e1(fs)
        _, predicts = out_e1.max(1)
        #print("len(predict)", len(predicts))
        correct = predicts.eq(Target_labels[:out_e1.shape[0]]).sum().item()
        # print(correct)
        if not correct == 2:
            delete_list += [i]
    print("delete_list:", len(delete_list))
    return np.delete(list_l1_soft, delete_list)#, high_confi_list[sorted_index[-1000:]]

if __name__ == '__main__':

    opt.logname = opt.logpath_data + 'log.tsv'

    if opt.dataset == 'GTSRB':
        #index
        high_confi_list = np.random.choice(np.arange(int(len(TrainDatasetGTSRBOri))),
                                                               size=20000, replace=False)
        #np.array(np.load(opt.logpath_data + 'high_con_index.npy', allow_pickle=True))[:15000]
        # sub_idxs_a = np.random.choice(high_confi_list, size=int(len(high_confi_list)/2), replace=False)
        # sub_idxs_b = high_confi_list[np.isin(high_confi_list, sub_idxs_a, invert=True)]
        #torch.save(opt.logpath + "sub_idxs.npy", high_confi_list)

        #valdataloader = TestDataloaderGTSRB
        traindataset = TrainDatasetGTSRBOri
        Dataset = Subset(traindataset, high_confi_list)
        dataloader = DataLoader(Dataset, batch_size=10, shuffle=True,
                                num_workers=opt.num_workers, drop_last=False)  # 15000

    elif opt.dataset == 'CelebA':
        high_confi_list = np.random.choice(np.arange(int(len(TrainDatasetCelebAOri))),
                                           size=20000, replace=False)
        traindataset = TrainDatasetCelebAOri
        Dataset = Subset(traindataset, high_confi_list)
        dataloader = DataLoader(Dataset, batch_size=10, shuffle=True,
                                num_workers=opt.num_workers, drop_last=False)  # 15000

    if not os.path.exists(opt.logpath_data):
        os.makedirs(opt.logpath_data)

    TriggerNet = Models.U_Net(output_ch=6*opt.output_class)
    TriggerNet = TriggerNet.cuda()
    optimizer_map = torch.optim.Adam(
        TriggerNet.parameters(), lr=opt.lr_optimizer_for_t)
    TriggerNet.load_state_dict(torch.load(opt.logpath_trigger + str(opt.trigger_name) + '.pth')['netP'])
    #Classifer.load_state_dict(torch.load(opt.logpath_trigger + str(50) + '.pth')['netC'])

    #get one model on a
    for i in range(1, 6):
        #the model is saved per 500 batches
        Train_for_subset(dataloader, epoch=i)

        # save classifier
        paras = {
            'netC': Classifer.state_dict(),
        }
        torch.save(paras, opt.logpath_data + str("{:.2f}".format(i+1)) + '.pth')

    #save the different loss/acc./confi. for different data, loss first
    #step 2
    Classifier_e1 = copy.deepcopy(Classifer)

    name_list = ["{:.2f}".format(x * 0.01) for x in list(np.arange(100, 590, 12.5))]
    print("name_list", name_list)
    list_l1_soft = []
    for i in range(len(name_list)-1):
        print("i", i)
        high_confi_list = np.asarray(list(set(high_confi_list)-set(list_l1_soft)))

        Dataset_temp = Subset(traindataset, high_confi_list)
        valdataloader = DataLoader(Dataset_temp, batch_size=opt.batch_size, shuffle=False,
                                   num_workers=opt.num_workers, drop_last=False)
        Classifier_e1.load_state_dict(torch.load(opt.logpath_data + str(name_list[i+1]) + '.pth')['netC']) #now
        Classifer.load_state_dict(torch.load(opt.logpath_data + str(name_list[i]) + '.pth')['netC']) #pre
        with torch.no_grad():
            list_l1_soft = choose_long_tail(valdataloader, high_confi_list)
        np.save(opt.logpath_data + 'list_l1({}).npy'.format(i+1), list_l1_soft)


    #np.save(opt.logpath_data + 'list_l1_s.npy', list_l1_low)
    #longtail_list = np.load(opt.logpath_data + 'longtail_index.npy', allow_pickle=True)
    #print("longtail_index,", len(longtail_list))
