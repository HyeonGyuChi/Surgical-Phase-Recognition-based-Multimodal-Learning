import torch.nn as nn
from core.model.loss_module.cb_loss import CBLoss, get_cb_loss
from core.model.loss_module.bs_loss import BalancedSoftmax, get_bs_loss
from core.model.loss_module.eqlv2 import EQLv2, get_eqlv2
# from core.model.loss_module.normsoftmax import NormSoftmaxLoss, get_normsoftmax


loss_dict = {
    'ce': nn.CrossEntropyLoss,
    'cb': get_cb_loss,
    'bs': get_bs_loss,
    'eqlv2': get_eqlv2,
    # 'normsm': get_normsoftmax,
}


def get_loss(args):    
    if 'ce' not in args.loss_fn:
        return loss_dict[args.loss_fn](args)
    else:
        return loss_dict[args.loss_fn]()
    
