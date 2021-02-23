import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

# Added
import time
import sys
from senet_scaled import *
from datahandler2 import *
from autoaugment import CIFAR10Policy, Cutout

def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total

        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        torch.save(state, 'ckpt.pth')
    return acc


def test(epoch):
    # print('\nEpoch: %d' % epoch)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total

    return acc


# gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data set
dataset = 'CIFAR10'
indx = 2
depth = float(sys.argv[indx])
depth = int(round(depth))
width = float(sys.argv[indx+1])
resolution = float(sys.argv[indx+2])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HPs
dataset = 'CIFAR10'

initial_image_size = 32
new_image_size = int(32 * resolution)
total_classes = 10
number_input_channels = 3

model = scaled_senet(depth, width, new_image_size)
model.to(device)

# print(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)     # , eta_min=1e-8

print('==> Preparing data..')

transform_train = transforms.Compose(
    [
        transforms.Resize((new_image_size, new_image_size)),
        transforms.RandomCrop(new_image_size, padding=4),  # resolution
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.Resize((new_image_size, new_image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)


# Subset of CIFAR-10
train_idx_dict = dict_indices(trainset)
test_idx_dict = dict_indices(testset)

best_valid_acc = []
best_train_acc = []

if device == 'cuda':
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True

for i in range(1):

    initialize(model)

    # Get 10% of dataset
    train_indices = get_indx_balanced_train_subset(train_idx_dict, i)
    print(len(train_indices))
    test_indices = get_indx_balanced_test_subset(test_idx_dict, i)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    # Sampler added to get a subset + this option was removed :
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, num_workers=4, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, num_workers=4, sampler=test_sampler)

    start_epoch = 0
    training_accuracies = []
    testing_accuracies = []
    t0 = time.time()
    execution_time = 0
    total_epochs = 0
    epoch = 0
    best_test_acc = 0

    while execution_time < 1000:

        tr_acc = train(epoch)
        training_accuracies.append(tr_acc)
        te_acc = test(epoch)
        testing_accuracies.append(te_acc)
        scheduler.step()
        execution_time = time.time() - t0
        if te_acc > best_test_acc:
            best_test_acc = te_acc
        lr = scheduler.get_last_lr()[0]

        print("Epoch {},  Train accuracy: {:.3f}, Val accuracy: {:.3f}, Best val acc: {:.3f}, LR: {:.3f}".format(epoch,
                                                                                                                 tr_acc,
                                                                                                                 te_acc,
                                                                                                                 best_test_acc,
                                                                                                                 lr))
        total_epochs += 1
        epoch += 1

    # best_valid_acc.append(max(testing_accuracies))
    # best_train_acc.append(max(training_accuracies))

print('Best valid acc', best_test_acc)

