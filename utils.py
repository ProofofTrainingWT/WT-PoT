from types import DynamicClassAttribute
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torch
from PIL import Image
import json
from config import opt
import csv
import torchvision
import torch.nn.functional as F
import time
import sys
import numpy as np

criterion = torch.nn.CrossEntropyLoss()
CLE = torch.nn.CrossEntropyLoss()
MAE = torch.nn.L1Loss()
BCE = torch.nn.BCELoss()

def prn_obj(obj):
    with open(opt.logname, 'w+') as f:
        f.write('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))
        f.write('\n')

Transform = transforms.Compose([
    transforms.Resize((opt.image_size)),
    transforms.CenterCrop(opt.image_size),
    transforms.ToTensor(),
])

def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

class GTSRB(Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train == 'Train':
            self.data_folder = os.path.join(opt.data_path, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        elif train == 'Eval':
            self.data_folder = os.path.join(opt.data_path, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

TrainDatasetGTSRBOri = GTSRB(opt, 'Train', Transform)
TrainDatasetGTSRB = GTSRB(opt, 'Train', Transform)
TestDatasetGTSRB = GTSRB(opt, 'Eval', Transform)

# subnet
train_subnet = True#True
test_subnet = True#False
# opt.logpath = './log/{}_{}_{}/'.format(opt.dataset,
#                                        opt.model_type, opt.tag)
if os.path.exists(opt.logpath + 'train_sub_idxs.npy'):
    gtsrb_train_sub_idxs = np.load(opt.logpath + 'train_sub_idxs.npy')
    gtsrb_test_sub_idxs = np.load(opt.logpath + 'test_sub_idxs.npy')
else:
    gtsrb_train_sub_idxs = np.random.choice(np.arange(int(len(TrainDatasetGTSRBOri))),
                                            size=int(len(TrainDatasetGTSRBOri) / 10), replace=False)

    gtsrb_test_sub_idxs = np.random.choice(np.arange(int(len(TestDatasetGTSRB))),
                                           size=int(len(TestDatasetGTSRB) / 10), replace=False)

if train_subnet:
    TrainDatasetGTSRB = Subset(TrainDatasetGTSRBOri, gtsrb_train_sub_idxs)
if test_subnet:
    TestDatasetGTSRB = Subset(TestDatasetGTSRB, gtsrb_test_sub_idxs)

TrainDataloaderGTSRB = DataLoader(TrainDatasetGTSRB, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers, drop_last=True)
TestDataloaderGTSRB = DataLoader(TestDatasetGTSRB, batch_size=opt.batch_size, shuffle=False,
                                 num_workers=opt.num_workers, drop_last=True)

class CelebA(Dataset):
    def __init__(self, opt, flag, transforms):
        super(CelebA, self).__init__()
        with open(os.path.join(opt.data_path, 'CelebA/CelebA_trainpaths'), 'r') as f:
            self.trainpaths = json.load(f)
        self.trainpaths_keys = list(self.trainpaths.keys())
        with open(os.path.join(opt.data_path, 'CelebA/CelebA_valpaths'), 'r') as f:
            self.evalpaths = json.load(f)
        self.evalpaths_keys = list(self.evalpaths.keys())
        self.flag = flag
        self.transforms = transforms

    def __getitem__(self, index):
        if self.flag == 'Train':
            imgpath = opt.data_path + 'CelebA/' + self.trainpaths_keys[index][2:]
            img = self.transforms(Image.open(imgpath))
            label = torch.tensor(self.trainpaths[self.trainpaths_keys[index]]).to(dtype=torch.long).squeeze()
        elif self.flag == 'Eval':
            imgpath = opt.data_path + 'CelebA/' + self.evalpaths_keys[index][2:]
            img = self.transforms(Image.open(imgpath))
            label = torch.tensor(self.evalpaths[self.evalpaths_keys[index]]).to(dtype=torch.long).squeeze()
        return img, label

    def __len__(self):
        if self.flag == 'Train':
            return len(self.trainpaths_keys)
        elif self.flag == 'Eval':
            return len(self.evalpaths_keys)

TrainDatasetCelebAOri = CelebA(opt, 'Train', Transform)
TestDatasetCelebAOri = CelebA(opt, 'Eval', Transform)

if os.path.exists(opt.logpath + 'celeb_train_sub_idxs.npy'):
    celeba_train_sub_idxs = np.load(opt.logpath + 'celeb_train_sub_idxs.npy')
    celeba_test_sub_idxs = np.load(opt.logpath + 'celeb_test_sub_idxs.npy')
else:
    celeba_train_sub_idxs = np.random.choice(np.arange(int(len(TrainDatasetCelebAOri))),
                                            size=int(len(TrainDatasetCelebAOri) / 4), replace=False)

    celeba_test_sub_idxs = np.random.choice(np.arange(int(len(TestDatasetCelebAOri))),
                                           size=int(len(TestDatasetCelebAOri) / 10), replace=False)

if train_subnet:
    TrainDatasetCelebA = Subset(TrainDatasetCelebAOri, celeba_train_sub_idxs)
else:
    TrainDatasetCelebA = TrainDatasetCelebAOri

if test_subnet:
    TestDatasetCelebA = Subset(TestDatasetCelebAOri, celeba_test_sub_idxs)
else:
    TestDatasetCelebA = TestDatasetCelebAOri

TrainDataloaderCelebA = DataLoader(TrainDatasetCelebA, batch_size=opt.batch_size, shuffle=True,
                                   num_workers=opt.num_workers, drop_last=True)
TestDataloaderCelebA = DataLoader(TestDatasetCelebA, batch_size=opt.batch_size, shuffle=False,
                                  num_workers=opt.num_workers, drop_last=True)

class Recorder():
    def __init__(self) -> None:
        self.train_loss = [0, 0]
        self.train_acc = [0, 0]
        self.count = [0, 0]
        self.probs = [0, 0]
        self.best_acc = 0
        self.currect_val_acc = 0
        self.patiencefor_increase = 0
        self.patiencefor_decrease = 0
        self.moving_normal_acc = []
        self.moving_poison_acc = []
        self.moving_l1norm = []

    def ac(self):
        self.train_loss = [0, 0]
        self.train_acc = [0, 0]
        self.count = [0, 0]
        self.probs = [0, 0]

    def half_ac(self):
        self.train_loss[1] = 0
        self.train_acc[1] = 0
        self.count[1] = 0

# term_width = #os.get_terminal_size() #_, term_width = os.popen("stty size", "r").read().split()
# term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time

# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.
#
#     cur_len = int(TOTAL_BAR_LENGTH * current / total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
#
#     sys.stdout.write(" [")
#     for i in range(cur_len):
#         sys.stdout.write("=")
#     sys.stdout.write(">")
#     for i in range(rest_len):
#         sys.stdout.write(".")
#     sys.stdout.write("]")
#
#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time
#
#     L = []
#     if msg:
#         L.append(" | " + msg)
#
#     msg = "".join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
#         sys.stdout.write(" ")
#
#     # Go back to the center of the bar.
#     for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
#         sys.stdout.write("\b")
#     sys.stdout.write(" %d/%d " % (current + 1, total))
#
#     if current < total - 1:
#         sys.stdout.write("\r")
#     else:
#         sys.stdout.write("\n")
#     sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f
