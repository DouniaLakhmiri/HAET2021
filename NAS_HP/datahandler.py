import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import random


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    random.shuffle(indices)
    return indices


def dict_indices(dataset):
    idx_classes = {}
    for i in range(10):
        idx_classes[i] = get_indices(dataset, i)
    return idx_classes


def get_indx_balanced_train_subset(dict_indices, k):
    indx_balanced_subset = []
    for i in range(10):
        indx_balanced_subset += dict_indices[i][k:k+500]
    return indx_balanced_subset


def get_indx_balanced_test_subset(dict_indices, k):
    indx_balanced_subset = []
    for i in range(10):
        indx_balanced_subset += dict_indices[i][k:k+100]
    return indx_balanced_subset


# def get_loader():
