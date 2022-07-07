import torch
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
        if 'gast' in args.dataset:
            weights = torch.Tensor([1.1219280406844807, 0.9681475764205234, 2.700095993716775, 
                                                0.9141240990478067, 0.2630276758636177, 1.3230899984498694, 
                                                0.7860581656049337, 1.0354759361569805, 0.9021204066360199, 
                                                0.5556540790991905, 4.018654414391012, 1.2661060557235002, 
                                                6.645501651139689, 0.7116979830072313, 1.0866479455282185, 
                                                1.7290175515393105, 0.4124589457336612, 1.2908398737972935, 
                                                1.9176686603241502, 0.48788876865370095, 8.124719629747148, 
                                                0.8676360619727526, 0.2317482728305169, 1.0, 0.6669441952076479, 
                                                0.8789846237615558, 1.495703685202791]).to(args.device)
            return loss_dict[args.loss_fn](weight=weights)
        else:
            return loss_dict[args.loss_fn]()
        
