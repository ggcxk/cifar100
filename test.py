
from resnet18 import ResNet18
from myutils import args,test_loader
import torch

# 测试函数
def testing(model):
    model.eval()
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        if args['cuda']:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    test_acc = correct / total
    model.train()
    return test_acc


def get_model_acc(method):
    # 模型
    model = ResNet18(num_classes=100)
    if args['cuda']:
        model = model.cuda()
    model.load_state_dict(torch.load('./mycheckpoints/CIFAR100_ResNet18_' + method + '.pth'))
    # 测试
    test_acc = testing(model)
    return test_acc


if __name__ == "__main__":
    method = 'baseline'
    acc = get_model_acc(method)
    print(acc)

    method = 'cutout'
    acc = get_model_acc(method)
    print(acc)

    method = 'mixup'
    acc = get_model_acc(method)
    print(acc)

    method = 'cutmix'
    acc = get_model_acc(method)
    print(acc)
