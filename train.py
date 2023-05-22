import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from resnet18 import ResNet18
from myutils import args,mixup_criterion,mixup_data,rand_bbox,CSVLogger,get_train_dataloader
from test import testing


def train_cutout(epoch,model,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        
        if args['cuda']:
            images = images.cuda()
            labels = labels.cuda()

        model.zero_grad()
        pred = model(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # 计算训练过程中的准确率
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        # 打印训练过程中的loss和acc
        progress_bar.set_postfix(xentropy='%.3f' % (xentropy_loss_avg / (i + 1)), acc='%.3f' % accuracy)

    return (xentropy_loss_avg / (i + 1)), accuracy


def train_mixup(epoch,model,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        if args['cuda']:
            inputs, targets = inputs.cuda(), targets.cuda()

        # mixup数据处理
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args['alpha'], args['cuda'])
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        model.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(xentropy='%.3f' % (train_loss / (batch_idx + 1)), acc='%.3f' % (correct / total))

    return (train_loss / (batch_idx + 1)), (correct / total).item() / 100


def train_cutmix(epoch,model,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        if args['cuda']:
            inputs, targets = inputs.cuda(), targets.cuda()

        r = np.random.rand(1)
        if args['alpha'] > 0 and r < args['cutmix_prob']: # 使用cutmix的概率
          # 生成mix的样本
            lam = np.random.beta(args['alpha'], args['alpha'])
            if args['cuda']:
                rand_index = torch.randperm(inputs.size()[0]).cuda()
            else:
                rand_index = torch.randperm(inputs.size()[0])
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # 调整lambda以与像素比匹配
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # 计算输出
            output = model(inputs)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            # 计算输出
            output = model(inputs)
            loss = criterion(output, targets)

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(xentropy='%.3f' % (train_loss / (batch_idx + 1)),
                acc='%.3f' % (correct / total))

    return (train_loss / (batch_idx + 1)), (correct / total).item()


def param_basedOn_method(method):
    if method == 'cutout':
        train = train_cutout
        writer = SummaryWriter('./runs/train_cutout') # 使用tensorboard进行可视化
    elif method == 'mixup':
        train = train_mixup
        writer = SummaryWriter('./runs/train_mixup')
    elif method == 'cutmix':
        train = train_cutmix
        writer = SummaryWriter('./runs/train_cutmix')
    elif method == 'baseline':
        train = train_cutout
        writer = SummaryWriter('./runs/train_baseline') 
    return train,writer



def train_save_model(method):
    # 模型
    model = ResNet18(num_classes=100)
    if args['cuda']:
        model = model.cuda()
    # 定义损失函数
    if args['cuda']:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'],
                    momentum=0.9, nesterov=True, weight_decay=5e-4)
    # 定义学习率优化
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.2)
    # 数据储存到csv文件
    try:
        os.makedirs('./runs')
    except:
        pass
    csv_logger = CSVLogger(fieldnames=['epoch', 'train_loss', 'train_acc', 'test_acc'],method=method)
    # 训练模型过程
    train,writer = param_basedOn_method(method)
    for epoch in range(1, args['epochs'] + 1):
        train_loss, train_acc = train(epoch,model,criterion,optimizer)
        test_acc = testing(model)
        tqdm.write('test_acc: %.3f' % test_acc)
        scheduler.step()
        row = {'epoch': str(epoch), 'train_loss':str(train_loss), 'train_acc': str(train_acc), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('train_acc', train_acc, global_step=epoch)
        writer.add_scalar('test_acc', test_acc, global_step=epoch)
    writer.close()
    # 保存模型
    try:
        os.makedirs('./mycheckpoints')
    except:
        pass
    torch.save(model.state_dict(), './mycheckpoints/CIFAR100_ResNet18_' + method + '.pth')
    csv_logger.close()


if __name__ == "__main__":
    args['cuda'] = torch.cuda.is_available()
    cudnn.benchmark = True

    torch.manual_seed(0)
    if args['cuda']:
        torch.cuda.manual_seed(0)

    # 分别执行下面四种方法
    method = 'baseline'
    train_loader = get_train_dataloader()
    train_save_model(method)

    method = 'cutmix'
    train_loader = get_train_dataloader()
    train_save_model(method)

    method = 'mixup'
    train_loader = get_train_dataloader()
    train_save_model(method)

    method = 'cutout'
    train_loader = get_train_dataloader(method)
    train_save_model(method)

