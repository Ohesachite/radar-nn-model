from __future__ import print_function
import datetime
import os
import time
import sys
import math
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils
from . import distance_utils
from . import radar

from scheduler import WarmupMultiStepLR

# Define measure types.
TYPE_MEASURE_GAN = 'GAN'  # Vanilla GAN.
TYPE_MEASURE_JSD = 'JSD'  # Jensen-Shannon divergence.
TYPE_MEASURE_KL = 'KL'    # KL-divergence.
TYPE_MEASURE_RKL = 'RKL'  # Reverse KL-divergence.
TYPE_MEASURE_H2 = 'H2'    # Squared Hellinger.
TYPE_MEASURE_W1 = 'W1'    # Wasserstein distance (1-Lipschitz).

# Define generator loss types.
TYPE_GENERATOR_LOSS_MM = 'MM'  # Minimax.
TYPE_GENERATOR_LOSS_NS = 'NS'  # Non-saturating.

TYPE_FUSION_OP_CAT = 'CAT' # concatenate
TYPE_FUSION_OP_POE = 'POE' # Product of Experts
TYPE_FUSION_OP_MOE = 'MOE' # Mixture of Experts

def compute_positive_indicator_matrix(anchors, matches, distance_fn, max_positive_distance):
    distance_matrix = distance_utils.compute_distance_matrix(anchors, matches, distance_fn)
    distance_matrix = (distance_matrix + torch.transpose(distance_matrix)) / 2.0
    positive_indicator_matrix = distance_matrix <= max_positive_distance
    positive_indicator_matrix = positive_indicator_matrix.type(torch.float32)
    return positive_indicator_matrix


def compute_positive_expectation(samples, measure, reduce_mean=False):
    if measure == TYPE_MEASURE_GAN:
        expectation = F.softplus(-samples)
    elif measure == TYPE_MEASURE_JSD:
        expectation = math.log(2.) - F.softplus(-samples)
    elif measure == TYPE_MEASURE_KL:
        expectation = samples

    if reduce_mean:
        return torch.mean(expectation)
    else:
        return expectation

def compute_negative_expectation(samples, measure, reduce_mean=False):
    if measure == TYPE_MEASURE_GAN:
        expectation = F.softplus(-samples) + samples
    elif measure == TYPE_MEASURE_JSD:
        expectation = F.softplus(-samples) + samples - math.log(2.)
    elif measure == TYPE_MEASURE_KL:
        expectation = torch.exp(samples - 1.)

    if reduce_mean:
        return torch.mean(expectation)
    else:
        return expectation


def compute_fenchel_dual_loss(local_features, global_features, measure, positive_indicator_matrix=None):
    device = local_features.get_device()
    batch_size, num_locals, local_feature_dim = local_features.shape
    _, num_globals, global_feature_dim = global_features.shape
    assert local_feature_dim == global_feature_dim, "Mismatch in feature sizes: {0} vs {1}".format(local_feature_dim, global_feature_dim)

    local_features = torch.reshape(local_features, (-1, local_feature_dim))
    global_features = torch.reshape(global_features, (global_feature_dim, -1))

    # FIXME: check whether it transpose automatically
    product = torch.matmul(local_features, global_features)
    product = torch.reshape(product, (batch_size, num_locals, batch_size, num_globals))

    if positive_indicator_matrix is None:
        positive_indicator_matrix = torch.eye(batch_size, dtype=torch.float32, device=device)
    negative_indicator_matrix = 1. - positive_indicator_matrix

    positive_expectation = compute_positive_expectation(product, measure, reduce_mean=False)
    negative_expectation = compute_negative_expectation(product, measure, reduce_mean=False)

    positive_expectation = torch.mean(positive_expectation, dim=(1,3))
    negative_expectation = torch.mean(negative_expectation, dim=(1,3))

    positive_expectation = torch.sum(positive_expectation * positive_indicator_matrix) / torch.maximum(
        torch.sum(positive_indicator_matrix), torch.tensor([1e-12], device=device))
    negative_expectation = torch.sum(negative_expectation * negative_indicator_matrix) / torch.maximum(
        torch.sum(negative_indicator_matrix), torch.tensor([1e-12], device=device))

    return negative_expectation - positive_expectation

def compute_representation_loss(inputs, targets, fusion_type, positive_indicator_matrix):
    point_cloud_ts = targets[0]
    feature_ts = targets[1]

    if fusion_type == TYPE_FUSION_OP_CAT:
        #FIXME need to check which dim to concat
        fusion_embeddings = torch.cat((point_cloud_ts, feature_ts), dim=1)
    elif fusion_type == TYPE_FUSION_OP_POE:
        fusion_embeddings = point_cloud_ts * feature_ts
    elif fusion_type == TYPE_FUSION_OP_MOE:
        fusion_embeddings = 0.5 * (point_cloud_ts + feature_ts)
    else:
        raise ValueError("Unknown fusion operation: {}".format(fusion_type))
    
    representation_loss = compute_fenchel_dual_loss(inputs, fusion_embeddings, TYPE_MEASURE_JSD, positive_indicator_matrix=positive_indicator_matrix)

    return representation_loss

def compute_positive_indicator_matrix(labels, device):
    batch_size = labels.size(dim=0)

    indicator_matrix = np.zeros((batch_size, batch_size))

    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if l1 == l2:
                indicator_matrix[i][j] = 1

    indicator_matrix = torch.tensor(indicator_matrix, device=device)

    return indicator_matrix