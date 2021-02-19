from datahandler2 import *
# from neural_net_motifs import *
# from thop import clever_format, profile
from autoaugment import CIFAR10Policy, Cutout

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
import matplotlib.pyplot as plt
from senet_train import *


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


torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HPs
dataset = 'CIFAR10'

# -1 -1 1 0 2 72 1 0.240000000000000 0 0.010000000000000 0.220000000000000 0.100000000000000


initial_image_size = 32
total_classes = 10
number_input_channels = 3


model = SENet18()
model.to(device)

print(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8)

print('==> Preparing data..')

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
        # transforms.RandomCrop(initial_image_size, padding=4),  # resolution
        transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),

        # transforms.RandomRotation(10),  # Rotates the image to a specified angel
        # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Performs actions like zooms, change shear angles.
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
        # Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
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
    test_indices = get_indx_balanced_test_subset(test_idx_dict, i)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    # Sampler added to get a subset + this option was removed :
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=4, sampler=train_sampler)

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

    lrs = []
    while execution_time < 600:

        tr_acc = train(epoch)
        training_accuracies.append(tr_acc)
        te_acc = test(epoch)
        testing_accuracies.append(te_acc)
        scheduler.step()
        execution_time = time.time() - t0
        if te_acc > best_test_acc:
            best_test_acc = te_acc
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        # print(lr)

        if epoch % 100 == 0:
        # if epoch == 100:
            # Reset lr
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        print("Epoch {},  Train accuracy: {:.3f}, Val accuracy: {:.3f}, Best val acc: {:.3f}, LR: {:.3f}".format(epoch,
                                                                                                                 tr_acc,
                                                                                                                 te_acc,
                                                                                                                 best_test_acc,
                                                                                                                 lr))
        total_epochs += 1
        epoch += 1

        # if (epoch == 10) and best_test_acc < 25:
        #     break

    # print(training_accuracies)
    # print(testing_accuracies)

    best_valid_acc.append(max(testing_accuracies))
    best_train_acc.append(max(training_accuracies))

