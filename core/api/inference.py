import os
import torch
import numpy as np
from tqdm import tqdm

from core.model import get_model, get_fusion_model
from core.dataset import get_dataset
from core.config.task_info import task_dict
from core.utils.metric import MetricHelper



class Predictor():
    def __init__(self, args):
        self.args = args
        self.setup()

    def setup(self):
        # Load dataset
        print('======= Load Dataset =======')
        self.train_loader, self.val_loader, ski_feature_num = get_dataset(self.args)
        self.set_class_weight()
        
        # Load model
        # Set device type [cpu or cuda]
        print('======= Load Model =======')
        if self.args.model == 'multi':
            self.model = get_fusion_model(self.args)
        else:
            self.model = get_model(self.args)
        
        # multi-task classifier setup
        if hasattr(self.model, 'set_classifiers'):
            self.model.set_classifiers(self.n_class_list)

        self.args.n_class_list = self.n_class_list

        self.model.to(self.args.device)

        # Load pre-trained model
        if self.args.restore_path is not None:
            print('======= Load Pretrained Model =======')
            
            states = torch.load(self.args.restore_path)
            if 'state_dict' in states:
                self.model.load_state_dict(states['state_dict'])
            else:     
                self.model.load_state_dict(states['model'])

        print('======= Set Metric Helper =======')
        self.metric_helper = MetricHelper(self.args)


    def set_class_weight(self):
        train_cnt_list, val_cnt_list = self.train_loader.dataset.class_cnt, self.val_loader.dataset.class_cnt
        total_cnt_list = []
        cls_weights = []
        
        for train_cnt, val_cnt in zip(train_cnt_list, val_cnt_list):
            total_cnt_list.append(np.zeros((len(train_cnt))))
            cls_weights.append(np.zeros((len(train_cnt))))
            
            total_cnt_list[-1] = train_cnt + val_cnt
            
        for idx in range(len(total_cnt_list)):
            if len(total_cnt_list[idx]):
                bot_sum = 0
                n_classes = len(total_cnt_list[idx])

                for idx2 in range(n_classes):
                    bot_sum += total_cnt_list[idx][idx2]

                for idx2 in range(n_classes):
                    cls_weights[idx][idx2] = bot_sum / (n_classes * total_cnt_list[idx][idx2])
                
                if idx < 2:
                    cls_weights[idx] = torch.Tensor(np.ones(len(total_cnt_list[idx]))).cuda()
                else:
                    cls_weights[idx] = torch.Tensor(total_cnt_list[idx]).cuda()
        
        self.args.class_cnt = total_cnt_list
        self.args.class_weights = cls_weights #self.train_loader.dataset.class_weights #cls_weights

        # task ??? class ??? ??????
        self.n_task = task_dict[self.args.dataset][self.args.task][0]
        self.n_class_list = []
        
        if self.args.task != 'all':
            if self.args.dataset == 'petraw' and self.args.task == 'action':
                self.n_class_list.append(task_dict[self.args.dataset][self.args.task][1] // 2)
                self.n_class_list.append(task_dict[self.args.dataset][self.args.task][1] // 2)
            else:
                self.n_class_list = [task_dict[self.args.dataset][self.args.task][1]]
        else:
            for task in task_dict[self.args.dataset]:
                if task != 'all':
                    if self.args.dataset == 'petraw' and task == 'action':
                        self.n_class_list.append(task_dict[self.args.dataset][task][1] // 2)
                        self.n_class_list.append(task_dict[self.args.dataset][task][1] // 2)
                    else:
                        self.n_class_list.append(task_dict[self.args.dataset][task][1])

    def forward(self, x, y):
        if self.args.model == 'multi':
            y_hat, _ = self.model(x)
        elif self.args.model == 'slowfast':
            y_hat = self.model.forward(x['video'], return_loss=False, infer_3d=True)
        else:
            y_hat = self.model(x)
        
        return y_hat

    @torch.no_grad()
    def inference(self):
        self.model.eval()

        for data in tqdm(self.val_loader, desc='[Inference Phase] : '):
            x, y = data
            if self.args.device == 'cuda':
                for k in x.keys():
                    x[k] = x[k].to(self.args.device)
                y = y.to(self.args.device)

            y_hat = self.forward(x, y)

            # if y.shape[-1] > 1:
            #     y = [y[..., yi].reshape(-1) for yi in range(y.shape[-1])]
            
            # cls_hat = []
            # for ti in range(len(self.n_class_list)):
            #     classes = torch.argmax(y_hat[ti], -1)
            #     cls_hat.append(classes.reshape(-1))
            cls_hat = []

            if self.args.dataset == 'petraw':
                if y.shape[-1] > 1:
                    y = [y[..., yi].reshape(-1) for yi in range(y.shape[-1])]

                for ti in range(len(self.n_class_list)):
                    classes = torch.argmax(y_hat[ti], -1)
                    cls_hat.append(classes.reshape(-1))
            else:
                cls_hat = torch.argmax(y_hat, -1).unsqueeze(0)
                y = y.unsqueeze(0)
            
            # break
            self.metric_helper.write_preds(cls_hat, y)
            
        # metric = self.metric_helper.calc_metric()
        metric = self.metric_helper.calc_metric2()

        return metric
        