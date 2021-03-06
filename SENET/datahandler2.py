import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import random
import numpy as np


def get_class_i_indices(y, i):
    y = np.array(y)
    pos_i = np.argwhere(y == i)
    pos_i = list(pos_i[:, 0])
    random.shuffle(pos_i)

    return pos_i


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
    # print(len(dict_indices[0]))
    indx_balanced_subset = []
    for i in range(10):
        p10_idx = len(dict_indices[i]) // 10
        # print(p10_idx)
        indx_balanced_subset += dict_indices[i][k:k + p10_idx]
    return indx_balanced_subset


def get_indx_balanced_test_subset(dict_indices, k):
    indx_balanced_subset = []
    for i in range(10):
        indx_balanced_subset += dict_indices[i][k:k + 100]
    return indx_balanced_subset
