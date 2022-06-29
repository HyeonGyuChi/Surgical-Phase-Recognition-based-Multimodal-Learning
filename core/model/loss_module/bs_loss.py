"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, freq):
        super(BalancedSoftmax, self).__init__()
        freq = torch.FloatTensor(freq).cuda()
        self.sample_per_class = freq

    def forward(self, logits, label, reduction='mean'):
        return balanced_softmax_loss(logits, label, self.sample_per_class, reduction)


def balanced_softmax_loss(logits, labels, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    # spc = sample_per_class.type_as(logits)
    spc = sample_per_class
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def create_loss(freq_path):
    print('Loading Balanced Softmax Loss.')
    return BalancedSoftmax(freq_path)


def get_bs_loss(config):
    loss_fns = []
    for i in range(len(config.class_weights)):
        class_cnt = config.class_weights[i].cpu().data.numpy()
        loss_fns.append( BalancedSoftmax(class_cnt) )
        
    return loss_fns