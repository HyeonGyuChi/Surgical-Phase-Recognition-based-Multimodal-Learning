import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleZConv(nn.Module):
    def __init__(self, n_modality, conv_sz_list, feat_dim):
        super().__init__()

        conv_list = []

        for conv_sz in conv_sz_list:
            conv_list.append(
                nn.Sequential(
                    nn.Conv1d(n_modality, n_modality, conv_sz, padding=conv_sz//2),
                    nn.ReLU(),
                    nn.BatchNorm1d(n_modality),
                    nn.Conv1d(n_modality, 1, conv_sz, padding=conv_sz//2),
                    nn.ReLU(),
                )
            )
        self.z_conv = nn.ModuleList(conv_list)
        self.to_z_emb = nn.Sequential(
                    nn.Linear(feat_dim * len(conv_sz_list), feat_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(feat_dim),
                    nn.Linear(feat_dim, feat_dim),
                    nn.ReLU(),
                )

    def forward(self, x):
        x_list = []
        for z_conv in self.z_conv:
            x_list.append(z_conv(x))
        
        x = torch.cat(x_list, 1).view(x.size(0), -1)
        
        return self.to_z_emb(x)


class ConvFusion(nn.Module):
    def __init__(self, config, n_modality, modality_size_list):
        super().__init__()
        self.config = config

        self.f_emb_sz = self.config.fusion_params['conv']['f_dim']
        self.z_emb_sz = self.config.fusion_params['conv']['z_dim']
        self.fusion_type = self.config.fusion_params['conv']['fusion_type']
        self.conv_sz = self.config.fusion_params['conv']['conv_sz']
        self.use_pairwise = self.config.fusion_params['conv']['use_pairwise']
        self.multi_scale = self.config.fusion_params['conv']['multi_scale']
        self.seq_size = self.config.clip_size
        self.n_modality = n_modality
        self.modality_size_list = modality_size_list

        self.Softmax = torch.nn.Softmax(dim=-1)

        to_embedding = []

        for idx in range(self.n_modality):
            to_embedding.append(
                nn.Linear(self.modality_size_list[idx], self.f_emb_sz)
            )

        self.to_f_emb = nn.ModuleList(to_embedding)
        
        to_embedding = []
            
        for idx in range(self.n_modality):
            to_embedding.append(
                nn.Sequential(
                    torch.nn.Linear(self.f_emb_sz, self.z_emb_sz),
                    torch.nn.BatchNorm1d(self.z_emb_sz),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.z_emb_sz, self.f_emb_sz),
                )
            )
        self.to_z_emb = nn.ModuleList(to_embedding)
        self.classifiers = []

        if self.fusion_type == 'a':
            self.dim_weights = nn.Parameter(torch.randn((self.n_modality, self.f_emb_sz)))
        elif self.fusion_type == 'b':            
            # self.z_conv = nn.Conv1d(self.n_modality, 1, self.conv_sz, padding=self.conv_sz//2)
            # self.z_conv = nn.Sequential(
            #     nn.Conv1d(self.n_modality, self.n_modality, self.conv_sz, padding=self.conv_sz//2),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(self.n_modality),
            #     nn.Conv1d(self.n_modality, 1, self.conv_sz, padding=self.conv_sz//2),
            #     nn.ReLU(),
            # )

            self.z_conv = MultiScaleZConv(self.n_modality, [3, 5, 7], self.f_emb_sz)

        elif self.fusion_type == 'c': # a + contrastive or b + contrastive
            if self.use_pairwise:
                self.loss_fn = nn.PairwiseDistance()
            else:
                self.loss_fn = nn.CosineSimilarity(dim=1)

            # self.z_conv = nn.Conv1d(self.n_modality, 1, self.conv_sz, padding=self.conv_sz//2)

            if self.multi_scale:
                self.z_conv = MultiScaleZConv(self.n_modality, [3, 5, 7], self.f_emb_sz)
            else:
                self.z_conv = nn.Sequential(
                    nn.Conv1d(self.n_modality, self.n_modality, self.conv_sz, padding=self.conv_sz//2),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.n_modality),
                    nn.Conv1d(self.n_modality, 1, self.conv_sz, padding=self.conv_sz//2),
                    nn.ReLU(),
                )
            

        elif self.fusion_type == 'd': # a + b
            self.dim_weights = nn.Parameter(torch.randn((self.n_modality, self.f_emb_sz)))
            self.z_conv = nn.Conv1d(self.n_modality, 1, self.conv_sz, padding=self.conv_sz//2)
            # self.z_conv = nn.Sequential(
            #     nn.Conv1d(self.n_modality, self.n_modality, self.conv_sz, padding=self.conv_sz//2),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(self.n_modality),
            #     nn.Conv1d(self.n_modality, 1, self.conv_sz, padding=self.conv_sz//2),
            #     nn.ReLU(),
            # )

            self.to_cls_emb = torch.nn.Linear(self.f_emb_sz * 2, self.f_emb_sz)

        elif self.fusion_type == 'e':
            to_embedding = []

            for idx in range(self.n_modality):
                to_embedding.append(
                    nn.Linear(self.f_emb_sz, self.f_emb_sz)
                )

            self.recon_emb = nn.ModuleList(to_embedding)
            self.loss_fn = nn.MSELoss()


            self.linear = nn.Linear(self.f_emb_sz*self.n_modality, self.f_emb_sz)

            self.z_conv = nn.Sequential(
                    nn.Conv1d(self.n_modality, self.n_modality, self.conv_sz, padding=self.conv_sz//2),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.n_modality),
                    nn.Conv1d(self.n_modality, 1, self.conv_sz, padding=self.conv_sz//2),
                    nn.ReLU(),
                )


    def set_classifiers(self, n_class_list):
        for n_class in n_class_list:
            self.classifiers.append(torch.nn.Linear(self.f_emb_sz, n_class))
            # self.classifiers.append(torch.nn.Linear(self.f_emb_sz, n_class * self.seq_size).to(self.device))
        
        self.classifiers = torch.nn.ModuleList(self.classifiers)


    def forward(self, features):
        f_emb_list = []
        z_emb_list = []
        loss_fuse = 0

        for idx in range(self.n_modality):
            f_emb_list.append(self.to_f_emb[idx](features[idx].view(-1, self.modality_size_list[idx])))
            z_emb_list.append(self.to_z_emb[idx](f_emb_list[idx]))

        # B x N x D
        z = torch.cat([_z.unsqueeze(1) for _z in z_emb_list], 1)

        if self.fusion_type == 'a':
            # B x D
            z2 = (z * self.dim_weights).sum(dim=1)
            loss_fuse = None

        elif self.fusion_type == 'b':
            # B x D
            z2 = self.z_conv(z)
            loss_fuse = None

        elif self.fusion_type == 'c':
            for i in range(self.n_modality):
                f = F.normalize(f_emb_list[i], 1)

                for j in range(self.n_modality):
                    if i != j:
                        _z = F.normalize(z_emb_list[i].detach(), 1)
                        if self.use_pairwise:
                            loss_fuse += self.loss_fn(f, _z)
                        else:
                            loss_fuse += (1-self.loss_fn(f, _z))

            # z2 = (z * self.dim_weights).sum(dim=1)
            z2 = self.z_conv(z)

        elif self.fusion_type == 'd': # a + b type
            tz1 = (z * self.dim_weights).sum(dim=1)
            tz2 = self.z_conv(z).squeeze()
            z2 = self.to_cls_emb(torch.cat([tz1, tz2], 1))
            loss_fuse = None

        elif self.fusion_type == 'e':
            # z_mat = torch.stack([z[bi,0:1, ].view(z.size(2), -1) * z[bi, 1:2, ] for bi in range(z.size(0))])
            # z2 = self.linear(torch.cat((torch.sum(z_mat, dim=1), torch.sum(z_mat, dim=2)), 1))
            # loss_fuse = None
            # z2 = self.linear(torch.cat(z_emb_list, 1))
            z2 = self.z_conv(z)

            loss_fuse = 0
            f_emb_list2 = []
            for idx in range(self.n_modality):
                f_emb_list2.append(self.recon_emb[idx](z2))
                loss_fuse += self.loss_fn(f_emb_list[idx], f_emb_list2[idx])

            # aux_outputs = []

            # for f_emb in f_emb_list:
            #     _outputs = []

            #     for ci in range(len(self.classifiers)):
            #         x = self.classifiers[ci](f_emb)
            #         x = x.view(z2.size(0), -1)
            #         out = self.Softmax(x)

            #         _outputs.append(out)
            #     aux_outputs.append(_outputs)

            # outputs = []
        
            # for ci in range(len(self.classifiers)):
            #     x = self.classifiers[ci](z2)
            #     # x = x.view(z2.size(0), self.seq_size, -1)
            #     x = x.view(z2.size(0), -1)

            #     out = self.Softmax(x)
            #     outputs.append(out)

            # return outputs, loss_fuse, aux_outputs
            

        outputs = []
        
        for ci in range(len(self.classifiers)):
            x = self.classifiers[ci](z2)
            # x = x.view(z2.size(0), self.seq_size, -1)
            x = x.view(z2.size(0), -1)

            out = self.Softmax(x)
            outputs.append(out)

        return outputs, loss_fuse