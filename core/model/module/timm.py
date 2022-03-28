import timm
import torch
import torch.nn as nn


def generate_timm_model(args):
    model = TIMM(args)
    
    return model


class TIMM(nn.Module):
    """
        SOTA model usage
        1. resnet18
        2. repvgg_b0
        3. mobilenetv3_large_100
        
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        arch_name = self.config.model
        
        model = timm.create_model(arch_name, pretrained=True)
        
        # help documents - https://fastai.github.io/timmdocs/create_model (how to use feature_extractor in timm)
        if self.config.model == 'swin_large_patch4_window7_224':
            self.feature_module = nn.Sequential(
                *list(model.children())[:-2],
            )
            self.gap = nn.AdaptiveAvgPool1d(1)
        else:
            self.feature_module = nn.Sequential(
                *list(model.children())[:-1]
            )
            
        self.classifier = nn.Linear(model.num_features, config.n_classes)
        
    def forward(self, x):
        features = self.feature_module(x)
        
        if self.config.model == 'swin_large_patch4_window7_224':
            features = self.gap(features.permute(0, 2, 1))
            
        output = self.classifier(features.view(x.size(0), -1))
        
        return output