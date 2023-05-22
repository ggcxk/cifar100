from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import csv
import numpy as np


# 定义参数
args = {"dataset": 'cifar100', 
      "model": 'resnet18',  
      "batch_size" : 128,
      "epochs" : 50,  
      "learning_rate": 0.1, 
      "n_holes": 1,
      "length": 16,
      'alpha':0.2,
      "cutmix_prob": 0.1
}


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_train_dataloader(method='baseline'):
    mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    # 训练集预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    if method == 'cutout':
        train_transform.transforms.append(Cutout(args['n_holes'], args['length']))
    cifar100_train_dataset = datasets.CIFAR100(root='./data/Cifar100',
                    train=True,
                    transform=train_transform,
                    download=True)
    cifar100_train_loader = DataLoader(dataset=cifar100_train_dataset,
                        batch_size=args['batch_size'],
                        shuffle=True,
                        pin_memory=True)
    return cifar100_train_loader

def get_test_dataloader():
    mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    # 测试集预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    cifar100_test_dataset = datasets.CIFAR100(root='./data/Cifar100',
                    train=False,
                    transform=test_transform,
                    download=True)
    cifar100_test_loader = DataLoader(dataset=cifar100_test_dataset,
                        batch_size=args['batch_size'],
                        shuffle=False,
                        pin_memory=True)
    return cifar100_test_loader

test_loader = get_test_dataloader()


class CSVLogger:
    def __init__(self, fieldnames, method ='baseline'):
        filename = './runs/CIFAR100_ResNet18_' + method + '.csv'
        self.csv_file = open(filename, 'a')
        writer = csv.writer(self.csv_file)
        writer.writerow([''])
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()






