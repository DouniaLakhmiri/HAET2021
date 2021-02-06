"""Train CIFAR-10 with PyTorch."""

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utils import progress_bar

# Added
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import sys
from neural_net import *
from statistics import mean, stdev
from datahandler import *


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/ckpt.pth')
    return acc


def test(epoch):
    print('\nEpoch: %d' % epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return acc


device = 'cuda' if torch.cuda.is_available() else 'cpu'

indx = 1

print('==> Reading Hyperparameters..')

# Architecture : A scaled version of ResNet18 ?
depth = float(sys.argv[indx])
width = float(sys.argv[indx + 1])
resolution = float(sys.argv[indx + 2])

# Batch size
batch_size = int(sys.argv[indx + 3])

# HPs
optimizer_choice = int(sys.argv[indx + 4])
arg1 = float(sys.argv[indx + 5])  # lr
arg2 = float(sys.argv[indx + 6])  # momentum
arg3 = float(sys.argv[indx + 7])  # weight decay
arg4 = float(sys.argv[indx + 8])

# Data
initial_image_size = int(32 * resolution)
total_classes = 10
number_input_channels = 3

net = NeuralNet(depth, width, initial_image_size)
print(net)
net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

try:
    if optimizer_choice == 1:
        optimizer = optim.SGD(net.parameters(), lr=arg1, momentum=arg2, weight_decay=arg3,
                              dampening=arg4)
    if optimizer_choice == 2:
        optimizer = optim.Adam(net.parameters(), lr=arg1, betas=(arg2, arg3), weight_decay=arg4)
    if optimizer_choice == 3:
        optimizer = optim.Adagrad(net.parameters(), lr=arg1, lr_decay=arg2, weight_decay=arg4,
                                  initial_accumulator_value=arg3)
    if optimizer_choice == 4:
        optimizer = optim.RMSprop(net.parameters(), lr=arg1, momentum=arg2, alpha=arg3, weight_decay=arg4)
except ValueError:
    print('optimizer got an empty list')
    exit(0)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-8)


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((initial_image_size, initial_image_size)),
    transforms.RandomCrop(initial_image_size, padding=4),  # resolution
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

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

for i in range(3):
    # Get 10% of dataset
    train_indices = get_indx_balanced_train_subset(train_idx_dict, i)
    test_indices = get_indx_balanced_test_subset(test_idx_dict, i)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    # Sampler added to get a subset + this option was removed :
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=2, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, sampler=test_sampler
    )

    start_epoch = 0
    training_accuracies = []
    testing_accuracies = []
    t0 = time.time()
    execution_time = 0
    total_epochs = 0
    epoch = 0

    while execution_time < 60:
        tr_acc = train(epoch)
        training_accuracies.append(tr_acc)
        te_acc = test(epoch)
        testing_accuracies.append(te_acc)
        scheduler.step()
        execution_time = time.time() - t0
        total_epochs += 1
        epoch += 1

    print(training_accuracies)
    print(testing_accuracies)
    best_valid_acc.append(max(testing_accuracies))
    best_train_acc.append(max(training_accuracies))

print('Mean best valid acc', mean(best_valid_acc))
print('Std best valid acc', stdev(best_valid_acc))

print('Mean best train acc', mean(best_train_acc))
print('Std best train acc', stdev(best_train_acc))

# plt.plot(range(total_epochs), training_accuracies, label='Training', color='r')
# plt.plot(range(total_epochs), testing_accuracies, label='Training', color='b')
# plt.show()
