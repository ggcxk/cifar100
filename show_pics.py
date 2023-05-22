import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from myutils import args,Cutout,rand_bbox




train_transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15), # 数据增强
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])
train_dataset = datasets.CIFAR100(root='./data/Cifar100',
                train=True,
                transform=train_transform,
                download=True)
train_loader = DataLoader(dataset=train_dataset,
                    batch_size=args['batch_size'],
                    shuffle=True,
                    pin_memory=True)


print("获取3张图片")
imgs, labels = next(iter(train_loader))
imgs=imgs[0:3]
print(imgs.shape)

# 经过数据增强后的原图
images = make_grid(imgs)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()

# 经过cutout处理
cut=Cutout(n_holes=1, length=16)
imgs1=[]
for i in range(3):
    out=cut(imgs[i])
    imgs1.append(out)
images = make_grid(imgs1)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()

# 经过mixup处理
lam = np.random.beta(0.2, 0.2)
batch_size = imgs.size()[0]
index = torch.randperm(batch_size)
imgs2 = lam * imgs + (1 - lam) * imgs[index, :]
images = make_grid(imgs2)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()

# 经过cutmix处理
r = np.random.rand(1)
lam = np.random.beta(0.2, 0.2)
rand_index = torch.randperm(imgs.size()[0])
bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
# 调整lambda以与像素比匹配
lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
images = make_grid(imgs)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()