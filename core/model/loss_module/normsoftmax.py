import math
import torch
import torch.nn as nn
from torch.nn import Parameter


class NormSoftmaxLoss(nn.Module):
    def __init__(self, dim, num_instances, temperature=0.05):
        super().__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1).to(embeddings.device)
        # embeddings : B x seq x dim / B x class x seq
        # norm_weight : class x dim (output x input)
        # linear : input * (input x output) + output

        prediction_logits = nn.functional.linear(embeddings, norm_weight)
        prediction_logits = prediction_logits.transpose(1,2)
        # print(embeddings.shape, norm_weight.shape, prediction_logits.shape, instance_targets.shape)

        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)

        return loss

    def inference(self, embeddings):
        norm_weight = nn.functional.normalize(self.weight, dim=1).to(embeddings.device)
        prediction_logits = nn.functional.linear(embeddings, norm_weight)
        prediction_logits = prediction_logits.transpose(1,2)

        return prediction_logits


class NormSoftmaxLoss2(nn.Module):
    def __init__(self, dim, num_instances, cw, temperature=0.05):
        super().__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss(weight=cw)

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1).to(embeddings.device)
        # embeddings : B x seq x dim / B x class x seq
        # norm_weight : class x dim (output x input)
        # linear : input * (input x output) + output

        prediction_logits = nn.functional.linear(embeddings, norm_weight)
        prediction_logits = prediction_logits.transpose(1,2)
        # print(embeddings.shape, norm_weight.shape, prediction_logits.shape, instance_targets.shape)

        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)

        return loss
    
    
    def get_normsoftmax(config):
        # TODO 수정 필요
        loss_fns = []
        
        dim = 256
        # if arch == 'lstm':
        #     dim = config['model'][arch]['linear_dim']
        # elif arch == 'resnet3d':
        #     dim = 512
        # elif arch == 'mmnet' and config['model'][arch]['use_auto_fusion']:
        #     dim = config['model'][arch]['emb_sz']
        # elif arch == 'mmnet' and config['model'][arch]['use_factorize']:
        #     dim = config['model'][arch]['f_emb_sz']
        
        for i in range(len(config.class_weights)):
            class_cnt = config.class_cnt[i]
            loss_fns.append(NormSoftmaxLoss(dim, class_cnt))
        
        return loss_fns