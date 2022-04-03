import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.models import get_model
from core.model.fusion_module import ConvFusion


class MMNet(nn.Module):
    def __init__(self, args: dict):
        super().__init__()

        self.args = args
        self.n_modality = len(self.args.model_params)
        self.seq_size = self.args.clip_size
        self.device = self.args.device
        
        # initialize backbones
        self.init_feature_models()
        
        # Multimodal fusion methods
        self.init_fusion_method()
        

    def init_feature_models(self):
        backbones = {}
        self.n_modality = 0
        self.modality_size_list = []
        
        # modality 별 pretrained model load
        for modality in self.args.model_params.keys():
            copy_args = copy.deepcopy(self.args)
            modality_info = self.args.model_params[modality]
            copy_args.model = modality_info['model']
            copy_args.input_size = modality_info['input_size']
            copy_args.restore_path = modality_info['restore_path']
            
            self.n_modality += 1
            self.modality_size_list.append(modality_info['feature_size'])

            model = get_model(copy_args)
            if copy_args.restore_path is not None:
                states = torch.load(copy_args.restore_path)
                
                model.load_state_dict(states['model'])
                
            for p in model.parameters():
                p.requires_grad = False
                
            backbones[modality] = model.to(self.device)
        
        setattr(self, 'backbones', backbones)

    def init_fusion_method(self):
        if self.args.fusion_type == 'conv':
            self.fusion_module = ConvFusion(self.args, self.n_modality, self.modality_size_list)
            
    def set_classifiers(self, n_class_list):
        self.fusion_module.set_classifiers(n_class_list, self.device)
        
    def forward(self, x):
        # torch.Size([8, 8, 14]) torch.Size([8, 3, 8, 224, 224])
        # B x seq x feat, B x ch x seq x H x W
        features = []
        for modality in x.keys():
            features.append(self.backbones[modality].get_feature(x, modality))
        
        return self.fusion_module(features)
        
        
def get_fusion_model(args):
    return MMNet(args)