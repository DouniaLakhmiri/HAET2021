from datahandler import *
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
from scalable_senet import *


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

# Data transformations

initial_image_size = 32
total_classes = 10
number_input_channels = 3

print('==> Preparing data..')
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(initial_image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Architecture
print('==> Building network architecture..')
model = scaled_senet(0.78, 0.67, initial_image_size)
model.to(device)
print(model)

if device == 'cuda':
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# Optimizer
print('==> Defining the Optimizer and its hyperparameters..')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.042, momentum=0.9, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-8)


# --------------------------------------------
# Dataset - Cifar10
# Plugin new dataset here
# --------------------------------------------

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
             'truck': 9}

y_train = trainset.targets
y_test = testset.targets

plane_indices = get_class_i_indices(y_train, classDict['plane'])
car_indices = get_class_i_indices(y_train, classDict['car'])
bird_indices = get_class_i_indices(y_train, classDict['bird'])
cat_indices = get_class_i_indices(y_train, classDict['cat'])
deer_indices = get_class_i_indices(y_train, classDict['deer'])
dog_indices = get_class_i_indices(y_train, classDict['dog'])
frog_indices = get_class_i_indices(y_train, classDict['frog'])
horse_indices = get_class_i_indices(y_train, classDict['horse'])
ship_indices = get_class_i_indices(y_train, classDict['ship'])
truck_indices = get_class_i_indices(y_train, classDict['truck'])

plane_indices_test = get_class_i_indices(y_test, classDict['plane'])
car_indices_test = get_class_i_indices(y_test, classDict['car'])
bird_indices_test = get_class_i_indices(y_test, classDict['bird'])
cat_indices_test = get_class_i_indices(y_test, classDict['cat'])
deer_indices_test = get_class_i_indices(y_test, classDict['deer'])
dog_indices_test = get_class_i_indices(y_test, classDict['dog'])
frog_indices_test = get_class_i_indices(y_test, classDict['frog'])
horse_indices_test = get_class_i_indices(y_test, classDict['horse'])
ship_indices_test = get_class_i_indices(y_test, classDict['ship'])
truck_indices_test = get_class_i_indices(y_test, classDict['truck'])

subset_indices_1 = plane_indices[0:500] + car_indices[0:500] + bird_indices[0:500] + cat_indices[0:500] + deer_indices[
                                                                                                          0:500] + dog_indices[
                                                                                                                   0:500] + frog_indices[
                                                                                                                            0:500] + horse_indices[
                                                                                                                                     0:500] + ship_indices[
                                                                                                                                              0:500] + truck_indices[
                                                                                                                                                       0:500]

subset_indices_test_1 = plane_indices_test[0:100] + car_indices_test[0:100] + bird_indices_test[
                                                                              0:100] + cat_indices_test[
                                                                                       0:100] + deer_indices_test[
                                                                                                0:100] + dog_indices_test[
                                                                                                         0:100] + frog_indices_test[
                                                                                                                  0:100] + horse_indices_test[
                                                                                                                           0:100] + ship_indices_test[
                                                                                                                                    0:100] + truck_indices_test[
                                                                                                                                             0:100]

trainset_1 = torch.utils.data.Subset(trainset, subset_indices_1)
testset_1 = torch.utils.data.Subset(testset, subset_indices_test_1)


# --------------------------------------------
# End of the dataset portion

# we need trainset_1 and testset_1 to define the trainloader and testloader
# --------------------------------------------

print('==> Model initialization..')
initialize(model)

# Sampler added to get a subset + this option was removed :
trainloader = torch.utils.data.DataLoader(
    trainset_1, batch_size=512, num_workers=2, shuffle=True)

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


while execution_time < 600:
    tr_acc = train(epoch)
    training_accuracies.append(tr_acc)
    te_acc = test(epoch)
    testing_accuracies.append(te_acc)
    scheduler.step()
    execution_time = time.time() - t0

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

    print(
        "Epoch {}, Execution time: {:.1f} Train accuracy: {:.3f}, Val accuracy: {:.3f}, Best val acc: {:.3f}, "
        "LR: {:.3f}".format(epoch, execution_time, tr_acc, te_acc, best_test_acc, lr))

    epoch += 1

print('Best valid acc', max(testing_accuracies))
print('Best train acc', max(training_accuracies))
