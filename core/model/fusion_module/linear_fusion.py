import torch
import torch.nn as nn
import torch.nn.functional as F



class LinearFusion(nn.Module):
    def __init__(self, config, n_modality, modality_size_list):
        super().__init__()

        self.config = config

        self.seq_size = self.config.clip_size
        self.n_modality = n_modality
        self.modality_size_list = modality_size_list
        self.f_emb_sz = self.config.fusion_params['conv']['f_dim']

        self.Softmax = torch.nn.Softmax(dim=-1)

        to_embedding = []

        for idx in range(self.n_modality):
            to_embedding.append(
                nn.Linear(self.modality_size_list[idx], self.f_emb_sz)
            )

        self.to_f_emb = nn.ModuleList(to_embedding)
        self.classifiers = []


    def set_classifiers(self, n_class_list):
        for n_class in n_class_list:
            self.classifiers.append(torch.nn.Linear(self.f_emb_sz, n_class))
            # self.classifiers.append(torch.nn.Linear(self.f_emb_sz, n_class * self.seq_size).to(self.device))
        
        self.classifiers = torch.nn.ModuleList(self.classifiers)

    
    def forward(self, features):
        f_emb_list = []
        loss_fuse = 0

        for idx in range(self.n_modality):
            f_emb_list.append(self.to_f_emb[idx](features[idx].view(-1, self.modality_size_list[idx])))

        # B x N x D
        z = torch.cat([f.unsqueeze(1) for f in f_emb_list], 1)

        outputs = []
        
        for ci in range(len(self.classifiers)):
            x = self.classifiers[ci](z)
            x = x.view(z.size(0), -1)

            out = self.Softmax(x)
            outputs.append(out)

        return outputs, loss_fuse