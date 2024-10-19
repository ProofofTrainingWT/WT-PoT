import random

from torch.utils.data import ConcatDataset, TensorDataset

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
import argparse

def random_mask(prob=opt.missing_proption):
    num = int(1/prob)
    if random.randint(1, num) == 1:
        return 0
    else:
        return 1

def Train_Warmup(dataloader, recoder, cp_recoder, epoch):  # features of the last epoch
    Classifer.train()  # Local model
    Classifer_counterpart.train()
    warmup_batch = len(dataloader)
    for index, (fs, labels) in enumerate(tqdm(dataloader)):
        fs = fs.cuda()
        labels = labels.cuda()
        optimizer_net.zero_grad()
        out, _ = Classifer(fs)  # f is features = torch.flatten(x, 1)

        loss = criterion(out, labels)
        loss.backward()
        optimizer_net.step()

        if random_mask() == 1:
            optimizer_net_counterpart.zero_grad()
            out_c, f_c = Classifer_counterpart(fs)
            loss_c = criterion(out_c, labels)
            loss_c.backward(retain_graph=True)
            optimizer_net_counterpart.step()
        else:
            loss_c = torch.tensor(0.0)

        with torch.no_grad():
            out_p, _ = Classifer(fs)
            _, predicts_p = out_p.max(1)
            out_p_c, _ = Classifer_counterpart(fs)
            _, predicts_p_c = out_p_c.max(1)

            if index < opt.overall_num-300:
                recoder.train_acc[0] += predicts_p.eq(labels).sum().item()
                recoder.count[0] += labels.shape[0]
                recoder.count[1] += labels.shape[0]
                recoder.train_loss[0] += loss.item()
                cp_recoder.train_acc[0] += predicts_p_c.eq(labels).sum().item()
                cp_recoder.train_loss[0] += loss_c.item()
                cp_recoder.count[0] += labels.shape[0]
                cp_recoder.count[1] = cp_recoder.count[0]
            else:
                recoder.train_acc[1] += predicts_p.eq(labels).sum().item()
                recoder.count[1] += labels.shape[0]
                recoder.count[0] += labels.shape[0]
                recoder.train_loss[1] += loss.item()
                out_prob = F.softmax(out_p, dim=0)
                recoder.probs[1] += torch.sum(out_prob[torch.arange(out_prob.shape[0]), labels])
                cp_recoder.train_acc[1] += predicts_p_c.eq(labels).sum().item()
                cp_recoder.train_loss[1] += loss_c.item()
                cp_recoder.count[1] += labels.shape[0]
                cp_recoder.count[0] += labels.shape[0]
                out_prob_c = F.softmax(out_p, dim=0)
                cp_recoder.probs[1] += torch.sum(out_prob_c[torch.arange(out_prob_c.shape[0]), labels])

        if (index+1) % opt.interval_batch == 0:
            with torch.no_grad():
                if opt.to_print == 'True':
                    print('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}'.format(
                        recoder.train_loss[0]/opt.interval_batch,
                        recoder.train_acc[0] * 100 / recoder.count[0],
                        recoder.train_loss[1]/opt.interval_batch,
                        (recoder.train_acc[1] / recoder.count[1]) * 100
                    ))
                    print(
                        'Train Counterpart model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Counterpart Poison acc:{:.2f}'.format(
                            cp_recoder.train_loss[0]/opt.interval_batch,
                            (cp_recoder.train_acc[0]) * 100 / recoder.count[0],
                            cp_recoder.train_loss[1]/opt.interval_batch,
                            (cp_recoder.train_acc[1] / cp_recoder.count[1]) * 100
                        ))
                if opt.to_save == 'True':
                    with open(opt.logname, 'a+') as f:
                        train_cols = ["{:.2f}".format(epoch + 1.0 * index / warmup_batch),
                                      "{:.4f}".format(recoder.train_loss[0] / opt.interval_batch),
                                      "{:.4f}".format((recoder.train_acc[0] / recoder.count[0]) * 100),
                                      "{:.4f}".format(recoder.train_loss[1] / opt.interval_batch),
                                      "{:.4f}".format((recoder.train_acc[1] / recoder.count[1]) * 100),
                                      "{:.4f}".format((recoder.probs[1] / recoder.count[1]) * 100),
                                      # cp_recoder.train_loss[0] / len(dataset),
                                      # "{:.4f}".format((cp_recoder.train_acc[0] / cp_recoder.count[0]) * 100),
                                      # cp_recoder.train_loss[1] / len(dataset),
                                      "{:.4f}".format((cp_recoder.train_acc[1] / cp_recoder.count[1]) * 100),
                                      "{:.4f}".format((cp_recoder.probs[1] / cp_recoder.count[1]) * 100)
                                      ]
                        f.write('\t'.join([str(c) for c in train_cols]) + '\n')
                recoder.ac()
                cp_recoder.ac()

    with torch.no_grad():
        recoder.ac()
        cp_recoder.ac()

def train_main(dataloader, wm_dataloader, recoder, cp_recoder, mode, epoch):
    if wm_dataloader is not None:
        wm_dataloader_iter = iter(wm_dataloader)

    for index, (fs, labels) in enumerate(tqdm(dataloader)):
        fs = fs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            if wm_dataloader is not None:
                try:
                    wm_data, wm_labels = next(wm_dataloader_iter)
                    imgs_input = torch.cat((fs, wm_data.cuda()), 0)
                    wm_labels = wm_labels.cuda()
                except:
                    # print("wm data is exhuasted")
                    # break
                    imgs_input = fs
                    wm_labels = []
            else:
                imgs_input = fs
                wm_labels = []

        optimizer_net.zero_grad()

        out, _ = Classifer(imgs_input)  # f is features = torch.flatten(x, 1)
        loss_ori = criterion(out[0: labels.shape[0], :], labels)
        if len(wm_labels) > 0:
            loss_p = criterion(out[labels.shape[0]::], wm_labels)  # backdoors
        else:
            loss_p = torch.tensor(0.0).cuda()
        loss = loss_ori + loss_p  # + loss_f * opt.a + Triggersl2norm * opt.b
        loss.backward()
        optimizer_net.step()

        if random_mask() == 1:
            optimizer_net_counterpart.zero_grad()
            out_c, f_c = Classifer_counterpart(imgs_input)
            loss_ori_c = criterion(out_c[0:labels.shape[0], :], labels)
            if len(wm_labels) > 0:
                loss_p_c = criterion(out_c[labels.shape[0]::], wm_labels)  # backdoors
            else:
                loss_p_c = torch.tensor(0.0).cuda()
            loss_c = loss_ori_c + loss_p_c  # + loss_f_c * opt.a + Triggersl2norm * opt.b
            loss_c.backward(retain_graph=True)
            optimizer_net_counterpart.step()
        else:
            loss_ori_c = torch.tensor(0.0)
            loss_p_c = torch.tensor(0.0)

        with torch.no_grad():
            out, _ = Classifer(imgs_input)
            out_ori = out[0:labels.shape[0], :]
            _, predicts_ori = out_ori.max(1)
            recoder.train_acc[0] += predicts_ori.eq(labels).sum().item()

            if len(wm_labels) > 0:
                out_p = out[labels.shape[0]::, :]
                _, predicts_p = out_p.max(1)
                recoder.train_acc[1] += predicts_p.eq(wm_labels).sum().item()
                recoder.count[1] += wm_labels.shape[0]
                out_prob = F.softmax(out_p, dim=0)
                recoder.probs[1] += torch.sum(out_prob[torch.arange(out_prob.shape[0]), wm_labels])
            else:
                recoder.count[1] += 1

            recoder.train_loss[0] += loss_ori.item()
            recoder.train_loss[1] += loss_p.item()
            recoder.count[0] += labels.shape[0]

            out_c, _ = Classifer_counterpart(imgs_input)
            out_ori_c = out_c[0:labels.shape[0], :]
            _, predicts_ori_c = out_ori_c.max(1)
            cp_recoder.train_acc[0] += predicts_ori_c.eq(labels).sum().item()
            if len(wm_labels) > 0:
                out_p_c = out_c[labels.shape[0]::, :]
                _, predicts_p_c = out_p_c.max(1)
                cp_recoder.train_acc[1] += predicts_p_c.eq(wm_labels).sum().item()
                cp_recoder.count[1] += wm_labels.shape[0]
                out_prob_c = F.softmax(out_p_c, dim=0)
                cp_recoder.probs[1] += torch.sum(out_prob_c[torch.arange(out_prob_c.shape[0]), wm_labels])
            else:
                cp_recoder.count[1] += 1e-2#wm_labels.shape[0]
            cp_recoder.train_loss[0] += loss_ori_c.item()
            cp_recoder.train_loss[1] += loss_p_c.item()
            cp_recoder.count[0] += labels.shape[0]


        if (index + 1) % opt.interval_batch == 0:
            with torch.no_grad():
                if opt.to_print == 'True':
                    print('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}'.format(
                        recoder.train_loss[0] / opt.interval_batch,
                        (recoder.train_acc[0] / recoder.count[0]) * 100,
                        recoder.train_loss[1] / opt.interval_batch,
                        (recoder.train_acc[1] / recoder.count[1]) * 100
                    ))
                    print(
                        'Train Counterpart model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Counterpart Poison acc:{:.2f}'.format(
                            cp_recoder.train_loss[0] / opt.interval_batch,
                            (cp_recoder.train_acc[0] / cp_recoder.count[0]) * 100,
                            cp_recoder.train_loss[1] / opt.interval_batch,
                            (cp_recoder.train_acc[1] / cp_recoder.count[1]) * 100
                        ))
                if opt.to_save == 'True':
                    with open(opt.logname, 'a+') as f:
                        train_cols = ["{:.2f}".format(epoch + 1.0 * index / len(dataloader) * ((3+37*mode)/(3+40+2.4)) +
                                                      mode*(3/(3+40+2.4))),
                                      recoder.train_loss[0] / opt.interval_batch,
                                      "{:.4f}".format((recoder.train_acc[0] / recoder.count[0]) * 100),
                                      # recoder.train_loss[1] / len(dataset),
                                      recoder.train_loss[1] / opt.interval_batch,
                                      "{:.4f}".format((recoder.train_acc[1] / recoder.count[1]) * 100),
                                      "{:.4f}".format((recoder.probs[1] / recoder.count[1]) * 100),
                                      # cp_recoder.train_loss[0] / len(dataset),
                                      # "{:.4f}".format((cp_recoder.train_acc[0] / cp_recoder.count[0]) * 100),
                                      # cp_recoder.train_loss[1] / len(dataset),
                                      "{:.4f}".format((cp_recoder.train_acc[1] / cp_recoder.count[1]) * 100),
                                      "{:.4f}".format((cp_recoder.probs[1] / cp_recoder.count[1]) * 100)
                                      ]
                        f.write('\t'.join([str(c) for c in train_cols]) + '\n')
                recoder.ac()
                cp_recoder.ac()

def Train_hard_wm(dataset, wm_dataset_temp, wm_soft_dataset_temp, cleanse_dataset_temp, remaining_cleanse_dataset_temp,
                  recoder, cp_recoder, epoch):
    Classifer.train()  # Local model
    Classifer_counterpart.train()
    overall_num = opt.overall_num - (opt.len_soft_wm_data + \
                  (opt.len_hard_wm_data-opt.hard_point_num * opt.wm_num * opt.wm_batch_size)) / (opt.batch_size + opt.wm_batch_size)
    len_hard_wm_data = opt.len_hard_wm_data
    point_num = opt.hard_point_num
    cl_num_temp = opt.cl_num * (opt.soft_point_num / opt.hard_point_num)
    # len_normal_data = np.floor(overall_num / point_num - (opt.wm_num + cl_num_temp)) * (
    #             opt.wm_batch_size + opt.batch_size)
    len_cleanse_normal_data = (opt.batch_size + opt.wm_batch_size - opt.cl_batch_size) * cl_num_temp

    for i in range(point_num):
        print('i', i)

        dataset_temp = Subset(dataset, list(range(int(i*(opt.batch_size * opt.wm_num + len_cleanse_normal_data)),
                                                  int(i*(opt.batch_size * opt.wm_num + len_cleanse_normal_data) +
                                                      opt.batch_size * opt.wm_num))))
        dataloader = DataLoader(dataset_temp,
                                batch_size=opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers)

        wm_dataloader = DataLoader(Subset(wm_dataset_temp, list(range(int(i * (opt.wm_batch_size * opt.wm_num)),
                                                                      int((i+1) * (opt.wm_batch_size * opt.wm_num))
                                                                      ))),
                                   batch_size=opt.wm_batch_size, shuffle=False,
                                   num_workers=opt.num_workers, drop_last=True)  # 3

        opt.interval_batch = int(opt.wm_num / 6)
        train_main(dataloader, wm_dataloader, recoder, cp_recoder, mode=0, epoch=epoch)

        cl_normal_dataset_temp = Subset(dataset,
                          list(range(int(i * (opt.batch_size * opt.wm_num + len_cleanse_normal_data) +
                                         opt.batch_size * opt.wm_num),
                                     int((i+1)*(opt.batch_size * opt.wm_num + len_cleanse_normal_data)))))

        cl_normal_dataset_temp = getShuffledtDataset(epoch, cl_normal_dataset_temp, f'hard_cl_no_{i}')
        dataloader = DataLoader(
            cl_normal_dataset_temp,
            batch_size=opt.batch_size + opt.wm_batch_size - opt.cl_batch_size, shuffle=False,
            num_workers=opt.num_workers)
        wm_dataloader = DataLoader(Subset(cleanse_dataset_temp, list(range(int(i * (cl_num_temp * opt.cl_batch_size)),
                                                           int((i+1)*(cl_num_temp * opt.cl_batch_size))))),
                                   batch_size=opt.cl_batch_size, shuffle=False,
                                   num_workers=opt.num_workers, drop_last=True)  # 3

        opt.interval_batch = 10
        train_main(dataloader, wm_dataloader, recoder, cp_recoder, mode=1, epoch=epoch)

    rest_dataset = Subset(dataset,
                          list(range(int(4*(opt.batch_size * opt.wm_num + len_cleanse_normal_data)),
                              int(len(dataset)))))
    # print("len_normal_data", len_normal_data)
    rest_dataset_temp = ConcatDataset([rest_dataset, remaining_cleanse_dataset_temp])
    rest_dataset_temp = getShuffledtDataset(epoch, rest_dataset_temp, f'hard_rest_{i}')
    dataloader = DataLoader(
        rest_dataset_temp,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers)  # 16
    remaining_hard_wm_dataset = Subset(wm_dataset_temp, list(range(int(4 * (opt.wm_batch_size * opt.wm_num)),
                                                                      int(len(wm_dataset_temp))
                                                                      )))
    wm_dataloader1 = DataLoader(
        ConcatDataset([remaining_hard_wm_dataset, wm_soft_dataset_temp]),
        batch_size=opt.wm_batch_size, shuffle=False,
        num_workers=opt.num_workers
    )
    opt.interval_batch = 10
    train_main(dataloader, wm_dataloader=wm_dataloader1, recoder=recoder, cp_recoder=cp_recoder, mode=0, epoch=epoch)

    with torch.no_grad():
        recoder.ac()
        cp_recoder.ac()

def Train_soft_wm(dataset, wm_dataset_temp, wm_hard_dataset_temp, cleanse_dataset_temp, remaining_cleanse_dataset_temp,
                  recoder, cp_recoder, epoch): #features of the last epoch
    Classifer.train() #Local model
    Classifer_counterpart.train()
    wm_num_temp = opt.soft_wm_num
    overall_num = opt.overall_num
    len_soft_wm_data = opt.len_soft_wm_data
    point_num = opt.soft_point_num
    # len_clean_part = np.floor(overall_num / point_num - (wm_num_temp + opt.cl_num)) * (
    #             opt.wm_batch_size + opt.batch_size)

    for i in range(point_num):
        len_cl_normal_set = (opt.batch_size + opt.wm_batch_size - opt.cl_batch_size) * opt.cl_num
        print('i', i)
        print("first_indices", int(i*(opt.batch_size * wm_num_temp + len_cl_normal_set)))
        print("second_indices", int(i*(opt.batch_size * wm_num_temp + len_cl_normal_set ) +
                                                      opt.batch_size * wm_num_temp))

        dataset_temp = Subset(dataset, list(range(int(i*(opt.batch_size * wm_num_temp + len_cl_normal_set)),
                                                  int(i*(opt.batch_size * wm_num_temp + len_cl_normal_set) +
                                                      opt.batch_size * wm_num_temp))))

        dataloader = DataLoader(dataset_temp,
                                batch_size=opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers)
        print("len_dataset_temp", len(dataset_temp))

        wm_dataloader = DataLoader(Subset(wm_dataset_temp, list(range(int(i*(opt.wm_batch_size * wm_num_temp)),
                                                                      int((i+1)*(opt.wm_batch_size * wm_num_temp))))),
                                   batch_size=opt.wm_batch_size, shuffle=False,
                                   num_workers=opt.num_workers, drop_last=True)  # 3
        print("len_wm_dataset_temp", len(list(range(int(i*(opt.wm_batch_size * wm_num_temp)),
                                                                      int((i+1)*(opt.wm_batch_size * wm_num_temp))))))

        opt.interval_batch = int(wm_num_temp / 6)
        train_main(dataloader, wm_dataloader, recoder, cp_recoder, mode=0, epoch=epoch)

        print("second_indices", int(i*(opt.batch_size * wm_num_temp + len_cl_normal_set) +
                                                        opt.batch_size * wm_num_temp))
        print("third_indices",
              int(i * (opt.batch_size * wm_num_temp + len_cl_normal_set) +
                  opt.batch_size * wm_num_temp + (opt.batch_size + opt.wm_batch_size - opt.cl_batch_size) * opt.cl_num))

        cl_normal_dataset_temp = Subset(dataset, list(range(int(i*(opt.batch_size * wm_num_temp + len_cl_normal_set) +
                                                        opt.batch_size * wm_num_temp),
                                              int(i*(opt.batch_size * wm_num_temp + len_cl_normal_set) +
                                                        opt.batch_size * wm_num_temp +
                                                  (opt.batch_size+opt.wm_batch_size-opt.cl_batch_size)*opt.cl_num))))

        cl_normal_dataset_temp = getShuffledtDataset(epoch, cl_normal_dataset_temp, f'soft_cl_no_{i}')
        dataloader = DataLoader(
            cl_normal_dataset_temp,
            batch_size=opt.batch_size+opt.wm_batch_size-opt.cl_batch_size, shuffle=False,
            num_workers=opt.num_workers)

        print("len_cl_normal_dataset_temp(w wm)", len(cl_normal_dataset_temp))

        wm_dataloader = DataLoader(Subset(cleanse_dataset_temp, list(range(int(i * (opt.cl_num * opt.cl_batch_size)),
                                                           int((i+1)*(opt.cl_num * opt.cl_batch_size))))),
                                   batch_size=opt.cl_batch_size, shuffle=False,
                                   num_workers=opt.num_workers, drop_last=True)  # 3

        print("cleanse_dataset_temp", len(list(range(int(i * (opt.cl_num * opt.cl_batch_size)),
                                                           int((i+1)*(opt.cl_num * opt.cl_batch_size))))))

        opt.interval_batch = 10
        train_main(dataloader, wm_dataloader, recoder, cp_recoder, mode=1, epoch=epoch)

    # rest_dataset = Subset(remaining_cleanse_dataset_temp,
    #                     list(range(4*(opt.cl_num * opt.cl_batch_size),
    #                                 len(dataset))))

    dataloader = DataLoader(
        remaining_cleanse_dataset_temp,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers) #14

    wm_dataloader = DataLoader(wm_hard_dataset_temp,
                                   batch_size=opt.wm_batch_size, shuffle=False,
                                   num_workers=opt.num_workers, drop_last=True)#need 400
    opt.interval_batch = 10
    train_main(dataloader, wm_dataloader, recoder, cp_recoder, mode=0, epoch=epoch)

    with torch.no_grad():
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
            test_cols = [
                Loss/len(dataset),
                         100*Correct/Tot,
                         recoder.best_acc]
            f.write('\t'.join([str(c) for c in test_cols]) + '\n')

def Eval_poison(dataset,feature_r, recoder):
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
            test_cols = [Loss/len(dataset), Acc, l1_norm, LF, Loss_c / len(dataset), Acc_c]
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
    F_out = F_out.permute(1, 0, 2)[:, list(range(opt.wm_classes)), :].reshape(-1,dim_f)
    F_out_counterpart = torch.stack(F_out_counterpart)
    F_out_counterpart = F_out_counterpart.permute(1, 0, 2)[:, list(range(opt.wm_classes)), :].reshape(-1, dim_f)
    return F_out.cuda(), F_out_counterpart.cuda()

def getDatasetWHardness():
    for i in range(1, 8):
        wm_sub_idxs0 = np.load(opt.logpath_data + 'list_l1({}).npy'.format(i))  # [:int(984)]#half
        print("wm_sun_dixs", len(wm_sub_idxs0))
        if i == 1:
            wm_sub_idxs = copy.deepcopy(wm_sub_idxs0)
        else:
            wm_sub_idxs = np.concatenate((wm_sub_idxs, wm_sub_idxs0), axis=0)

    soft_wmdataset = Subset(TrainDatasetGTSRBOri, wm_sub_idxs[:int(opt.len_soft_wm_data)])  # 4*320
    # wmdataset_0 = Subset(TrainDatasetGTSRBOri, wm_sub_idxs[:int(2560)]) #16*160
    for i in range(10, 15):  # (10,15)for the GTSRB_resnet18_Ours
        wm_sub_idxs2 = np.load(opt.logpath_data + 'list_l1({}).npy'.format(i))
        print('wm_sub_idxs', len(wm_sub_idxs2))
        if i == 10:
            wm_sub_idxs_temp = copy.deepcopy(wm_sub_idxs2)
        else:
            wm_sub_idxs_temp = np.concatenate((wm_sub_idxs_temp, wm_sub_idxs2), axis=0)
    hard_wmdataset = Subset(TrainDatasetGTSRBOri, wm_sub_idxs_temp[-int(opt.len_hard_wm_data):])
    return soft_wmdataset, hard_wmdataset, np.concatenate((wm_sub_idxs[:int(opt.len_soft_wm_data)],
                                                           wm_sub_idxs_temp[-int(opt.len_hard_wm_data):]), axis=0)

def getShuffledtDataset(epoch, dataset, suffix):
    if not osp.exists(opt.logpath_set_idx):
        os.makedirs(opt.logpath_set_idx)

    if not osp.exists(opt.logpath_set_idx + f'{epoch}_order_{suffix}.npy'):
        set_idxs = np.random.choice(np.arange(len(dataset)),
                                    size=int(len(dataset)), replace=False)
        np.save(opt.logpath_set_idx + f'{epoch}_order_{suffix}.npy', set_idxs)
    else:
        set_idxs = np.load(opt.logpath_set_idx + f'{epoch}_order_{suffix}.npy')

    return Subset(dataset, set_idxs)

class BeanDataset(Dataset):
    def __init__(self, new_data, new_labels):
        self.new_data = new_data
        self.new_labels = new_labels

    def __len__(self):
        return len(self.new_data)

    def __getitem__(self, index):
        return self.new_data[index], self.new_labels[index].numpy()

def construct_wm_cl_dataset(soft_wmdataset, hard_wmdataset, cleanse_dataset):
    soft_wmdataloader = DataLoader(soft_wmdataset, batch_size=10, shuffle=False, drop_last=False)
    hard_wmdataloader = DataLoader(hard_wmdataset, batch_size=10, shuffle=False, drop_last=False)
    cleansedataloader = DataLoader(cleanse_dataset, batch_size=10, shuffle=False, drop_last=False)

    Target_labels = torch.stack([i * torch.ones(1) for i in list(range(opt.wm_classes))]).expand(
        int(opt.wm_classes), opt.batch_size + opt.wm_batch_size).permute(1, 0).to(dtype=torch.long,
                                                                                  device='cuda')  # .reshape(-1, 1).squeeze().to(dtype=torch.long,device='cuda')
    with torch.no_grad():
        wm_dataset_data_list = []
        wm_dataset_label_list = []
        hard_wm_dataset_data_list = []
        hard_wm_dataset_label_list = []
        cl_dataset_data_list = []
        cl_dataset_label_list = []
        for fs, _ in soft_wmdataloader:
            fs = fs.cuda()
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
            fs_poison = fs + Triggers
            imgs_input = fs_poison  # = torch.cat((imgs, fs_poison), 0)  # just join together
            target_labels = torch.gather(Target_labels, 1,
                                         random_index.view(Triggers.shape[0], 1).expand(-1, 1)).squeeze()
            wm_dataset_data_list.append(imgs_input.cpu())
            wm_dataset_label_list.append(target_labels.cpu())
        new_data = torch.cat(wm_dataset_data_list)
        new_labels = torch.cat(wm_dataset_label_list)
        soft_wmdataset_temp = BeanDataset(new_data, new_labels)

        for fs, _ in hard_wmdataloader:
            fs = fs.cuda()
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
            fs_poison = fs + Triggers
            imgs_input = fs_poison  # = torch.cat((imgs, fs_poison), 0)  # just join together
            target_labels = torch.gather(Target_labels, 1,
                                         random_index.view(Triggers.shape[0], 1).expand(-1, 1)).squeeze()
            hard_wm_dataset_data_list.append(imgs_input.cpu())
            hard_wm_dataset_label_list.append(target_labels.cpu())
        new_data = torch.cat(hard_wm_dataset_data_list)
        new_labels = torch.cat(hard_wm_dataset_label_list).to(torch.int64)
        hard_wmdataset_temp = BeanDataset(new_data, new_labels)

        for fs, labels in cleansedataloader:
            fs = fs.cuda()
            Triggers = TriggerNet(fs)
            Triggers = EmbbedNet(Triggers[:, : 3 * int(opt.wm_classes), :, :],
                                 Triggers[:, 3 * opt.output_class: 3 * opt.output_class + 3 * int(opt.wm_classes),
                                 :,
                                 :])
            Triggers = (Triggers) / 255
            Triggers = Triggers.reshape(-1, int(opt.wm_classes), 3, opt.image_size, opt.image_size)
            random_index = torch.randint(int(opt.wm_classes), size=(Triggers.shape[0],)).cuda()
            Triggers = torch.gather(Triggers, 1,
                                    random_index.view(Triggers.shape[0], 1, 1, 1, 1).expand(-1, -1,
                                                                                            Triggers.shape[2],
                                                                                            Triggers.shape[3],
                                                                                            Triggers.shape[4]))
            Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)

            revers_trigger = ReverseNet(fs)
            revers_trigger = EmbbedNet(revers_trigger[:, 0:3, :, :], revers_trigger[:, 3:6, :, :])
            revers_trigger = (revers_trigger) / 255
            revers_trigger = revers_trigger.reshape(-1, 3, opt.image_size, opt.image_size)

            fs_cleanse = fs + Triggers + revers_trigger
            fs_cleanse = torch.clip(fs_cleanse, min=0, max=1)
            cl_dataset_data_list.append(fs_cleanse.cpu())
            cl_dataset_label_list.append(labels)
        new_data = torch.cat(cl_dataset_data_list)
        new_labels = torch.cat(cl_dataset_label_list).to(torch.int64)
        cleanse_dataset_temp = BeanDataset(new_data, new_labels)

    return soft_wmdataset_temp, hard_wmdataset_temp, cleanse_dataset_temp

def construct_wm_dataset(wmdataset_list=None):#soft_wmdataset, hard_wmdataset, wm_test_dataset
    Target_labels = torch.stack([i * torch.ones(1) for i in list(range(opt.wm_classes))]).expand(
        int(opt.wm_classes), 10).permute(1, 0).to(dtype=torch.long, device='cuda')  # .reshape(-1, 1).squeeze().to(dtype=torch.long,device='cuda')

    if len(wmdataset_list) > 0:
        final_wmdataset_list = []
        for dataset in wmdataset_list:
            with torch.no_grad():
                wm_dataset_data_list = []
                wm_dataset_label_list = []
                wm_dataloader = DataLoader(dataset,  batch_size=10, shuffle=False, drop_last=False)
                for fs, _ in wm_dataloader:
                    fs = fs.cuda()
                    Triggers = TriggerNet(fs)
                    Triggers = EmbbedNet(Triggers[:, : 3 * int(opt.wm_classes), :, :],
                                         Triggers[:,
                                         3 * opt.output_class: 3 * opt.output_class + 3 * int(opt.wm_classes), :,
                                         :])
                    Triggers = (Triggers) / 255
                    Triggers = Triggers.reshape(-1, int(opt.wm_classes), 3, opt.image_size, opt.image_size)

                    random_index = torch.randint(int(opt.wm_classes), size=(Triggers.shape[0],)).cuda()
                    Triggers = torch.gather(Triggers, 1,
                                            random_index.view(Triggers.shape[0], 1, 1, 1, 1).expand(-1, -1,
                                                                                                    Triggers.shape[2],
                                                                                                    Triggers.shape[3],
                                                                                                    Triggers.shape[4]))
                    Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
                    fs_poison = fs + Triggers
                    imgs_input = fs_poison  # = torch.cat((imgs, fs_poison), 0)  # just join together
                    target_labels = torch.gather(Target_labels, 1,
                                                 random_index.view(Triggers.shape[0], 1).expand(-1, 1)).squeeze()

                    wm_dataset_data_list.append(imgs_input.cpu())
                    wm_dataset_label_list.append(target_labels.cpu())
                new_data = torch.cat(wm_dataset_data_list)
                new_labels = torch.cat(wm_dataset_label_list)
                final_wmdataset_list.append(BeanDataset(new_data, new_labels))
    else:
        return

    return tuple(final_wmdataset_list)

def construct_cl_dataset(cleanse_dataset_list=None):
    if len(cleanse_dataset_list) > 0:
        with torch.no_grad():
            cleanse_dataset_temp_list = []
            for cleanse_dataset in cleanse_dataset_list:
                cl_dataset_data_list = []
                cl_dataset_label_list = []
                cleansedataloader = DataLoader(cleanse_dataset, batch_size=10, shuffle=False, drop_last=False)
                for fs, labels in cleansedataloader:
                    fs = fs.cuda()
                    Triggers = TriggerNet(fs)
                    Triggers = EmbbedNet(Triggers[:, : 3 * int(opt.wm_classes), :, :],
                                         Triggers[:,
                                         3 * opt.output_class: 3 * opt.output_class + 3 * int(opt.wm_classes),
                                         :, :])
                    Triggers = (Triggers) / 255
                    Triggers = Triggers.reshape(-1, int(opt.wm_classes), 3, opt.image_size, opt.image_size)
                    random_index = torch.randint(int(opt.wm_classes), size=(Triggers.shape[0],)).cuda()
                    Triggers = torch.gather(Triggers, 1,
                                            random_index.view(Triggers.shape[0], 1, 1, 1, 1).expand(-1, -1,
                                                                                                    Triggers.shape[2],
                                                                                                    Triggers.shape[3],
                                                                                                    Triggers.shape[4]))
                    Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)

                    revers_trigger = ReverseNet(fs)
                    revers_trigger = EmbbedNet(revers_trigger[:, 0:3, :, :], revers_trigger[:, 3:6, :, :])
                    revers_trigger = (revers_trigger) / 255
                    revers_trigger = revers_trigger.reshape(-1, 3, opt.image_size, opt.image_size)

                    fs_cleanse = fs + Triggers + revers_trigger
                    fs_cleanse = torch.clip(fs_cleanse, min=0, max=1)
                    cl_dataset_data_list.append(fs_cleanse.cpu())
                    cl_dataset_label_list.append(labels)
                new_data = torch.cat(cl_dataset_data_list)
                new_labels = torch.cat(cl_dataset_label_list).to(torch.int64)
                cleanse_dataset_temp_list.append(BeanDataset(new_data, new_labels))
    else:
        return

    return tuple(cleanse_dataset_temp_list)

import sys
if __name__ == '__main__':

    EmbbedNet = Embbed()
    EmbbedNet = EmbbedNet.cuda()

    model_dict = {
        'alexnet': Models.alexnet(num_classes=opt.num_class),
        'vgg11': Models.vgg11(num_classes=opt.num_class),
        'vgg11_bn': Models.vgg11_bn(num_classes=opt.num_class),
        'vgg13_bn': Models.vgg13_bn(num_classes=opt.num_class),

        'vgg16_bn': Models.vgg16_bn(num_classes=opt.num_class),
        'resnet18': Models.resnet18(num_classes=opt.num_class),
        'resnet34': Models.resnet34(num_classes=opt.num_class),
        'resnet50': Models.resnet18(num_classes=opt.num_class),

        'densenet161': Models.densenet161(num_classes=opt.num_class),
        'densenet121': Models.densenet121(num_classes=opt.num_class),
        'densenet201': Models.densenet201(num_classes=opt.num_class),
        'densenet169': Models.densenet169(num_classes=opt.num_class),

        'wide_resnet50_2': Models.wide_resnet50_2(num_classes=opt.num_class),
        'wide_resnet28_10': Models.wide_resnet28_10(num_classes=opt.num_class),
        'wide_resnet101_2': Models.wide_resnet101_2(num_classes=opt.num_class)

    }

    ReverseNet = Models.U_Net(output_ch=6)
    ReverseNet = ReverseNet.cuda()
    ReverseNet.load_state_dict(torch.load(opt.logpath_clean + str(opt.cleanse_trigger_name) + '.pth')['netR'])

    if not os.path.exists(opt.logpath):
        os.makedirs(opt.logpath)

    opt.cfgname = osp.join(opt.logpath, 'config.txt')

    with open(opt.cfgname, 'w+') as f:
        #f.write("Try to use smaller clean data to converge faster; " + '\n')
        f.write(f'no warmup' + '\n')
        for arg, value in opt.__dict__.items():
            f.write(f"{arg}:{value}" + '\n')

    recoder_train = Recorder()
    recoder_val = Recorder()
    recoder_train_cp = Recorder()
    recoder_val_cp = Recorder()

    Classifer = model_dict[opt.pretrained_name]
    Classifer = Classifer.cuda()
    # it's a pre-trained model
    # Classifer.load_state_dict(torch.load(opt.logpath_clean + str(opt.pretrained_name) + '.pth')['netC'])
    Classifer_counterpart = copy.deepcopy(Classifer.cuda())
    # Classifer.load_state_dict(torch.load(opt.logpath + str(391) + '.pth')['netC'])
    # Classifer_counterpart.load_state_dict(torch.load(opt.logpath + str(391) + '.pth')['netCc'])

    optimizer_net = torch.optim.Adam(
        Classifer.parameters(), lr=opt.lr_optimizer_for_c, weight_decay=opt.weight_decay)
    optimizer_net_counterpart = torch.optim.Adam(
        Classifer_counterpart.parameters(), lr=opt.lr_optimizer_for_c, weight_decay=opt.weight_decay)

    opt.logname = osp.join(opt.logpath, 'log_{}-{}_{}({}).tsv'.format(opt.wm_num,
                                                                        opt.cl_num, opt.pretrained_name,
                                                                        opt.log_index))
    if osp.exists(opt.logname):
        print("may cover existing experiments! change log_index")

    with open(opt.logname, 'w+') as f:
        f.write('start \n')
        columns = ['epoch',
                   'clean_loss(Train)',
                   'clean_acc(Train)',
                   'poison_loss(Train)',
                   'poison_acc(Train)',
                   'poison_probs(Train)',
                   # 'clean_loss(Train Counterpart)',
                   # 'clean_acc(Train, cp)',
                   # 'poison_loss',
                   'poison_acc(Train cp)',
                   'poison_probs(Train cp)'
                   # 'normal_loss(Eval)', 'test_acc', 'best_acc'
                   ]
        f.write('\t'.join(columns) + '\n')

    TriggerNet = Models.U_Net(output_ch=6 * opt.output_class)
    TriggerNet = TriggerNet.cuda()
    TriggerNet.load_state_dict(torch.load(opt.logpath_trigger + str(opt.trigger_name) + '.pth')['netP'])
    if opt.dataset == 'GTSRB':
        # traindataloader = TrainDataloaderGTSRB
        soft_wmdataset, hard_wmdataset, selected_indices = getDatasetWHardness()

        all_indices = set(range(len(TrainDatasetGTSRBOri)))
        remaining_indices = list(all_indices - set(selected_indices))
        len_remaining = len(remaining_indices)
        split_idx = opt.cl_batch_size * opt.cl_num * opt.soft_point_num  #

        cleanse_indices = list(np.random.choice(np.asarray(remaining_indices), size=split_idx, replace=False))
        third_indices = list(set(remaining_indices) - set(cleanse_indices))
        normal_indices = list(np.random.choice(np.asarray(third_indices),
                                               size=opt.soft_point_num * (
                                                       opt.batch_size * opt.soft_wm_num + (opt.batch_size +
                                                                                           opt.wm_batch_size - opt.cl_batch_size) * opt.cl_num),
                                               replace=False))
        remaining_cleanse_indices = list(set(third_indices) - set(normal_indices))
        print("traindatasetGTSRBOri", len(TrainDatasetGTSRBOri))
        traindataset = Subset(TrainDatasetGTSRBOri, normal_indices)
        print("traindataset", len(traindataset))
        cleanse_dataset = Subset(TrainDatasetGTSRBOri, cleanse_indices)
        remaining_cleanse_dataset = Subset(TrainDatasetGTSRBOri, remaining_cleanse_indices)
        valdataloader = TestDataloaderGTSRB
        # soft_wmdataset, hard_wmdataset, cleanse_dataset = construct_wm_cl_dataset(soft_wmdataset,
        #                                                                           hard_wmdataset, cleanse_dataset)
        soft_wmdataset, hard_wmdataset = construct_wm_dataset([soft_wmdataset, hard_wmdataset])
        cleanse_dataset, remaining_cleanse_dataset = construct_cl_dataset(
            [cleanse_dataset, remaining_cleanse_dataset])

    elif opt.dataset == 'CelebA':
        traindataloader = TrainDataloaderCelebA
        valdataloader = TestDataloaderCelebA

    # if not os.path.exists(opt.logpath + 'train_sub_idxs.npy'):
    #    np.save(opt.logpath + 'train_sub_idxs.npy', train_sub_idxs)
    #    np.save(opt.logpath + 'test_sub_idxs.npy', test_sub_idxs)
    epochs = [1, 21]

    for epoch in range(epochs[0], epochs[1]):
        start = time()
        print('epoch:{}'.format(epoch))

        if epoch == 1:
            opt.interval_batch = 10
            cleanse_dataset_temp = getShuffledtDataset(epoch, cleanse_dataset, 'cl')
            wmdataset_temp = getShuffledtDataset(epoch, ConcatDataset([soft_wmdataset, hard_wmdataset]),
                                                 'wm')  # 240 240
            traindataset_temp = getShuffledtDataset(epoch, traindataset, 'nm')
            # dataset = ConcatDataset([traindataset_temp, cleanse_dataset_temp, wmdataset_temp])
            # print("traindataset_temp.len", len(traindataset_temp))
            # print("cleanse_dataset_temp.len", len(cleanse_dataset_temp))#2693
            # print("wmdataset_temp.len", len(wmdataset_temp))
            #
            # print("dataset.len", len(dataset))#40585
            dataloader = DataLoader(traindataset_temp, batch_size=16, shuffle=False,
                                    num_workers=opt.num_workers, drop_last=True)
            Train_Warmup(dataloader, recoder_train, recoder_train_cp, epoch)

        elif epoch > 1 and epoch < 6:
            opt.interval_batch = 10
            traindataset_temp = getShuffledtDataset(epoch, traindataset, 'soft_trainset')
            wm_dataset_temp = getShuffledtDataset(epoch, soft_wmdataset, 'soft_wm')
            wm_hard_dataset_temp = getShuffledtDataset(epoch, hard_wmdataset, 'hard_wm')
            cleanse_dataset_temp = getShuffledtDataset(epoch, cleanse_dataset, 'cl')
            remaining_cleanse_dataset_temp = getShuffledtDataset(epoch, remaining_cleanse_dataset, 'rcl')
            print("wm_soft_data", len(wm_dataset_temp))
            Train_soft_wm(traindataset_temp, wm_dataset_temp, wm_hard_dataset_temp, cleanse_dataset_temp,
                          remaining_cleanse_dataset_temp,
                          recoder_train, recoder_train_cp, epoch)

        elif epoch >= 6:
            opt.interval_batch = 10
            traindataset_temp = getShuffledtDataset(epoch, traindataset, 'hard_trainset')
            wm_dataset_temp = getShuffledtDataset(epoch, hard_wmdataset, 'hard_wm')
            wm_soft_dataset_temp = getShuffledtDataset(epoch, soft_wmdataset, 'soft_wm')
            cleanse_dataset_temp = getShuffledtDataset(epoch, cleanse_dataset, 'cl')
            remaining_cleanse_dataset_temp = getShuffledtDataset(epoch, remaining_cleanse_dataset, 'rcl')
            Train_hard_wm(traindataset_temp, wm_dataset_temp, wm_soft_dataset_temp, cleanse_dataset_temp,
                          remaining_cleanse_dataset_temp,
                          recoder_train, recoder_train_cp, epoch)
        end = time()

        print('cost time:{:.2f}s'.format(end - start))
