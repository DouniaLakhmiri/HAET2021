from datahandler2 import *
from neural_net_motifs import *
# from thop import clever_format, profile

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
# from utils import progress_bar

# Added
import time
import sys
from statistics import mean, stdev


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
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

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
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # print(epoch, test_loss, acc)

    return acc


torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HPs
dataset = 'CIFAR10'
# num_conv_layers = int(sys.argv[2])  # 0

num_full_layers = 5
list_blocks = num_full_layers * [-1]

for i in range(num_full_layers):
    list_blocks[i] = int(sys.argv[2 + i])
print(list_blocks)
indx = len(list_blocks)

batch_size_index = 7
batch_size = int(sys.argv[batch_size_index])
print(batch_size)

optimizer_choice = int(sys.argv[batch_size_index + 1])
arg1 = float(sys.argv[batch_size_index + 2])
arg2 = float(sys.argv[batch_size_index + 3])
arg3 = float(sys.argv[batch_size_index + 4])
arg4 = float(sys.argv[batch_size_index + 5])

dropout = float(sys.argv[batch_size_index + 6])

initial_image_size = 32
total_classes = 10
number_input_channels = 3

model = NeuralNet(list_blocks, initial_image_size, total_classes, number_input_channels, dropout)
model.to(device)

criterion = nn.CrossEntropyLoss()
try:
    if optimizer_choice == 1:
        optimizer = optim.SGD(model.parameters(), lr=arg1, momentum=arg2, weight_decay=arg3,
                              dampening=arg4)
    if optimizer_choice == 2:
        optimizer = optim.Adam(model.parameters(), lr=arg1, betas=(arg2, arg3), weight_decay=arg4)
    if optimizer_choice == 3:
        optimizer = optim.Adagrad(model.parameters(), lr=arg1, lr_decay=arg2, weight_decay=arg4,
                                  initial_accumulator_value=arg3)
    if optimizer_choice == 4:
        optimizer = optim.RMSprop(model.parameters(), lr=arg1, momentum=arg2, alpha=arg3, weight_decay=arg4)
except ValueError:
    print('optimizer got an empty list')
    exit(0)

# optimizer = optim.SGD(model.parameters(), lr=0.03)
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

if device == 'cuda':
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True

for i in range(3):

    # initialize(model)

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
    best_test_acc = 0

    while execution_time < 600:

        tr_acc = train(epoch)
        training_accuracies.append(tr_acc)
        te_acc = test(epoch)
        testing_accuracies.append(te_acc)
        scheduler.step()
        execution_time = time.time() - t0
        if te_acc > best_test_acc:
            best_test_acc = te_acc
        print("Epoch {},  Train accuracy: {:.3f}, Val accuracy: {:.3f}, Best val acc: {:.3f}".format(epoch,
                                                                                             tr_acc,
                                                                                             te_acc, best_test_acc))
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
