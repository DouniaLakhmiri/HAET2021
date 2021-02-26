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
from senet_scaled import *


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


initial_image_size = 32
new_image_size = int(32 * 1.02)
total_classes = 10
number_input_channels = 3


# Architecture
print('==> Building network architecture.')
model = scaled_senet(0.78, 0.67, new_image_size)
# model = scaled_senet(0.83, 0.54, new_image_size)
model.to(device)
# print(model)


criterion = nn.CrossEntropyLoss()

print('==> Preparing data..')
transform_train = transforms.Compose(
[
    transforms.Resize((new_image_size, new_image_size)),
    transforms.RandomCrop(new_image_size, padding=4),  # fill parameter needs torchvision installed from source
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

classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,'truck': 9}

y_train = trainset.targets
y_test = testset.targets

plane_indices = get_class_i_indices(y_train,classDict['plane'])
car_indices = get_class_i_indices(y_train,classDict['car'])
bird_indices = get_class_i_indices(y_train,classDict['bird'])
cat_indices = get_class_i_indices(y_train,classDict['cat'])
deer_indices = get_class_i_indices(y_train,classDict['deer'])
dog_indices = get_class_i_indices(y_train,classDict['dog'])
frog_indices = get_class_i_indices(y_train,classDict['frog'])
horse_indices = get_class_i_indices(y_train,classDict['horse'])
ship_indices = get_class_i_indices(y_train,classDict['ship'])
truck_indices = get_class_i_indices(y_train,classDict['truck'])

plane_indices_test = get_class_i_indices(y_test,classDict['plane'])
car_indices_test = get_class_i_indices(y_test,classDict['car'])
bird_indices_test = get_class_i_indices(y_test,classDict['bird'])
cat_indices_test = get_class_i_indices(y_test,classDict['cat'])
deer_indices_test = get_class_i_indices(y_test,classDict['deer'])
dog_indices_test = get_class_i_indices(y_test,classDict['dog'])
frog_indices_test = get_class_i_indices(y_test,classDict['frog'])
horse_indices_test = get_class_i_indices(y_test,classDict['horse'])
ship_indices_test = get_class_i_indices(y_test,classDict['ship'])
truck_indices_test = get_class_i_indices(y_test,classDict['truck'])

subset_indices_1 = plane_indices[0:500] + car_indices[0:500] + bird_indices[0:500] + cat_indices[0:500] + deer_indices[0:500] + dog_indices[0:500] + frog_indices[0:500] + horse_indices[0:500] + ship_indices[0:500]+ truck_indices[0:500]

subset_indices_test_1 = plane_indices_test[0:100] + car_indices_test[0:100] + bird_indices_test[0:100] + cat_indices_test[0:100] + deer_indices_test[0:100] + dog_indices_test[0:100] + frog_indices_test[0:100] + horse_indices_test[0:100] + ship_indices_test[0:100] + truck_indices_test[0:100]


trainset_1 = torch.utils.data.Subset(trainset, subset_indices_1)
testset_1 = torch.utils.data.Subset(testset, subset_indices_test_1)

best_valid_acc = []
best_train_acc = []

if device == 'cuda':
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True

print('==> Reading Hyperparameters..')

indx = 1

# Index of the cifar10 data subset
indexDataSeed = int(sys.argv[indx])
# set the random seed
random.seed(indexDataSeed)


#1    "batch_size": {"_type":"choice", "_value": [128, 256, 512]}, OR  "batch_size": {"_type":"choice", "_value": [1024, 2048]}
# Code for small batch sizes
batch_size_index = int(sys.argv[indx + 1])
if batch_size_index == 1:
    batch_size_arg = 128
elif batch_size_index == 2:
    batch_size_arg = 256
elif batch_size_index == 3:
    batch_size_arg = 512
else:
    print('Batch size index not registered')
    exit(0)
## Alternate code for LARGE batch sizes
#batch_size_index = int(sys.argv[indx + 1])
#if batch_size_index == 1:
#    batch_size_arg = 1024
#elif batch_size_index == 2:
#    batch_size_arg = 2048
#else:
#    print('Batch size index not registered')
#    exit(0)
    

#2    "weight_decay": {"_type":"choice", "_value": [0, 0.00005, 0.0005,  0.005, 0.05, 0.5]},
weight_decay_index = float(sys.argv[indx + 2])  # weight decay
if weight_decay_index == 1:
    weight_decay_arg = 0.0
elif weight_decay_index == 2:
    weight_decay_arg = 0.00005
elif weight_decay_index == 3:
    weight_decay_arg = 0.0005
elif weight_decay_index == 4:
    weight_decay_arg = 0.005
elif weight_decay_index == 5:
    weight_decay_arg = 0.05
elif weight_decay_index == 6:
    weight_decay_arg = 0.5
else:
    print('weight decay index not registered')
    exit(0)

#3    "lr":{"_type":"loguniform", "_value":[0.6 ; 1E-3]}, mapping lr = 10^xNomad + scaling for each optimizer
lr_Nomad = float(sys.argv[indx + 3]) # lr
lr_arg = pow(10,lr_Nomad)

#4     "optimizer":{"_type":"choice", "_value":["Adadelta"(1->lr_arg*10) , "Adagrad" (0.01->lr_arg/10) , "SGD" (0.1->lr_arg), "Adam" (0.001->lr_arg/100), "Adamw" (0.01->lr_arg/10),  "Adamax" (0.002->lr_arg/50),  "ASGD"]}
optimizer_choice = int(sys.argv[indx + 4])
try:
    if optimizer_choice == 1:
        optimizer = optim.Adadelta(model.parameters(), lr=lr_arg*10,  weight_decay=weight_decay_arg)
    if optimizer_choice == 2:
        optimizer = optim.Adagrad(model.parameters(), lr=lr_arg/10,  weight_decay=weight_decay_arg)
    if optimizer_choice == 3:
        optimizer = optim.SGD(model.parameters(), lr=lr_arg, momentum = 0.9, weight_decay=weight_decay_arg)
    if optimizer_choice == 4:
        optimizer = optim.Adam(model.parameters(), lr=lr_arg/100, weight_decay=weight_decay_arg)
    if optimizer_choice == 5:
        optimizer = optim.AdamW(model.parameters(), lr=lr_arg/100, weight_decay=weight_decay_arg)
    if optimizer_choice == 6:
        optimizer = optim.Adamax(model.parameters(), lr=lr_arg/50, weight_decay=weight_decay_arg)
    if optimizer_choice == 7:
        optimizer = optim.ASGD(model.parameters(), lr=lr_arg/10, weight_decay=weight_decay_arg)
except ValueError:
    print('optimizer got an empty list')
    exit(0)
    
print(' Done setting parameters.')

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-8)     # , eta_min=1e-8

initialize(model)

# Sampler added to get a subset + this option was removed :
trainloader = torch.utils.data.DataLoader(
    trainset_1, batch_size=batch_size_arg, num_workers=2, shuffle=True)

testloader = torch.utils.data.DataLoader(
    testset_1, batch_size=128, shuffle=False)


start_epoch = 0
training_accuracies = []
testing_accuracies = []
t0 = time.time()
execution_time = 0
total_epochs = 0
epoch = 0
best_test_acc = 0

lrs = []

print('Index seed:',indexDataSeed,'batch_size: ',batch_size_arg,' weight_decay: ',weight_decay_arg,' lr: ',lr_arg,' optimizer_choice: ',optimizer_choice)

 while execution_time < 600:
# while execution_time < 1000:

    tr_acc = train(epoch)
    training_accuracies.append(tr_acc)
    te_acc = test(epoch)
    testing_accuracies.append(te_acc)
    scheduler.step()
    execution_time = time.time() - t0
    
    # For challenge submission (not really needed for HPO)
    if te_acc > best_test_acc:
        best_test_acc = te_acc
        print('Saving checkpoint..')
        state = {
            'net': model.state_dict(),
            'acc': best_test_acc,
            'epoch': epoch,
        }
        torch.save(state, 'ckpt.pth')
    lr = scheduler.get_last_lr()[0]
    lrs.append(lr)
    # print(lr)

    print("Epoch {}, Execution time: {:.1f} Train accuracy: {:.3f}, Val accuracy: {:.3f}, Best val acc: {:.3f}, LR: {:.3f}".format(epoch,execution_time,tr_acc,te_acc,best_test_acc,lr))
    
    # Maybe adjusted
    if ((int(epoch)>30) and (best_test_acc<=50.0)):
        print('Stop after 30 epochs because best_test_acc<=50')
        break
    
    total_epochs += 1
    epoch += 1
    
print('Best valid acc',max(testing_accuracies))
print('Best train acc',max(training_accuracies))


