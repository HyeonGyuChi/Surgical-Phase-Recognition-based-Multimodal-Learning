import os
import torch
import torch.nn as nn
import numpy as np
import json
import natsort
import time
from glob import glob
from tqdm import tqdm

from core.model import get_model, get_fusion_model, get_loss, configure_optimizer
from core.dataset import get_dataset
from core.config.task_info import task_dict
from core.utils.metric import MetricHelper


class Trainer():
    # Multimodal Trainer
    def __init__(self, config: dict):
        self.config = config
        self.setup()
        
    def setup(self):
        self.current_epoch = 1
        
        # make log directory
        self.config.save_path += '/{}-cb-loss-ski_ocr2_test'.format(self.config.model)
        # self.config.save_path += '/{}-cb-loss-ski_ocr-type-c-pair'.format(self.config.model)
        os.makedirs(self.config.save_path, exist_ok=True)
                

        param_dict = vars(self.config)
        with open(self.config.save_path + '/params.json', 'w') as f:
            json.dump(param_dict, f, indent=4)


        # Load dataset
        print('======= Load Dataset =======')
        self.train_loader, self.val_loader = get_dataset(self.config)
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
        
        self.config.class_cnt = total_cnt_list
        self.config.class_weights = cls_weights #self.train_loader.dataset.class_weights #cls_weights
        
        # Load model
        print('======= Load Model =======')
        if self.config.model == 'multi':
            self.model = get_fusion_model(self.config)
        else:
            self.model = get_model(self.config)
        
        self.n_task = task_dict[self.config.dataset][self.config.task][0]
        self.n_class_list = []
        
        if self.config.task != 'all':
            if self.config.dataset == 'petraw' and self.config.task == 'action':
                self.n_class_list.append(task_dict[self.config.dataset][self.config.task][1] // 2)
                self.n_class_list.append(task_dict[self.config.dataset][self.config.task][1] // 2)
            else:
                self.n_class_list = [task_dict[self.config.dataset][self.config.task][1]]
        else:
            for task in task_dict[self.config.dataset]:
                if task != 'all':
                    if self.config.dataset == 'petraw' and task == 'action':
                        self.n_class_list.append(task_dict[self.config.dataset][task][1] // 2)
                        self.n_class_list.append(task_dict[self.config.dataset][task][1] // 2)
                    else:
                        self.n_class_list.append(task_dict[self.config.dataset][task][1])
        
        self.model.set_classifiers(self.n_class_list)
        self.config.n_class_list = self.n_class_list
        
        # Load loss function
        print('======= Load Loss Function =======')
        self.loss_fn = get_loss(self.config)
        
        # Load Optimizer, scheduler
        print('======= Load Optimizers =======')
        self.optimizer, self.scheduler = configure_optimizer(self.config, self.model)
    
        # Set device type [cpu or cuda]
        self.model = self.model.to(self.config.device)
        
        # Set multi-gpu
        if self.config.num_gpus > 1:
            print('======= Set Multi-GPU =======')
            self.model = nn.DataParallel(self.model)
            # self.model = nn.DataParallel(self.model, 
            #                             device_ids=list(range(config['model']['params']['n_gpus'])))
            
            
        # Load pre-trained model
        if self.config.restore_path is not None:
            print('======= Load Pretrained Model =======')
            
            states = torch.load(self.config.restore_path)           
            self.model.load_state_dict(states['model'])
            
            if self.config.resume:
                self.optimizer.load_state_dict(states['optimizer'])
                self.scheduler.load_state_dict(states['scheduler'])
                self.current_epoch = states['epoch']
        
        print('======= Set Metric Helper =======')
        self.metric_helper = MetricHelper(self.config)
        
    def fit(self):
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.config.max_epoch + 1):
            self.current_epoch = epoch
            self.metric_helper.update_epoch(epoch)
            
            # train phase
            self.train()
            
            # validation phase
            self.valid()
        
    def train(self):
        self.model.train()
        
        for data in tqdm(self.train_loader, desc='[Epoch {} - Train Phase] : '.format(self.current_epoch)):
            self.optimizer.zero_grad()
            
            x, y = data
            if self.config.device == 'cuda':
                for k in x.keys():
                    x[k] = x[k].to(self.config.device)
                y = y.to(self.config.device)
                
            y_hat, loss = self.forward(x, y)
            
            self.metric_helper.write_loss(loss.item(), 'train')
            
            loss.backward()
            self.optimizer.step()
            
        self.metric_helper.update_loss('train')
    
    @torch.no_grad()
    def valid(self):
        self.model.eval()

        for data in tqdm(self.val_loader, desc='[Epoch {} - Validation Phase] : '.format(self.current_epoch)):
            x, y = data
            if self.config.device == 'cuda':
                for k in x.keys():
                    x[k] = x[k].to(self.config.device)
                y = y.to(self.config.device)
                
            y_hat, loss = self.forward(x, y)
            
            self.metric_helper.write_loss(loss.item(), 'valid')
            
            if y.shape[-1] > 1:
                y = [y[..., yi].reshape(-1) for yi in range(y.shape[-1])]
            
            cls_hat = []
            for ti in range(len(self.n_class_list)):
                classes = torch.argmax(y_hat[ti], -1)
                cls_hat.append(classes.reshape(-1))
            
            self.metric_helper.write_preds(cls_hat, y)
            
        self.metric_helper.update_loss('valid')
        self.metric_helper.save_loss_pic()
        metric = self.metric_helper.calc_metric()
        
        if self.config.lr_scheduler == 'reduced':
            self.scheduler.step(self.metric_helper.get_loss('valid'))
        else:
            self.scheduler.step()
        
        if self.metric_helper.update_best_metric(metric):
            self.save_checkpoint()
        
    def forward(self, x, y):
        if self.config.model == 'multi':
            y_hat, fuse_loss = self.model(x)
            
            loss = self.calc_loss(y_hat, y)
            if fuse_loss is not None:
                loss += torch.mean(fuse_loss)
        else:
            y_hat = self.model(x)
        
            loss = self.calc_loss(y_hat, y)
        
        return y_hat, loss 
        
    def calc_loss(self, y_hat, y):
        # y_hat : N_task x (B x seq X C)
        # y : B x seq x (N_task classes)        
        loss = 0
        loss_div_cnt = 0

        if 'ce' in self.config.loss_fn:
            for ti in range(len(self.n_class_list)):
                for seq in range(y.shape[1]):
                    loss += self.loss_fn(y_hat[ti][:, seq, ], y[:, seq, ti])
                loss_div_cnt += 1
        else: # cb, bs, eqlv2
            for ti in range(len(self.n_class_list)):
                for seq in range(y.shape[1]):
                    loss += self.loss_fn[ti](y_hat[ti][:, seq, :], y[:, seq, ti])
                loss_div_cnt += 1
                    
            if self.config.use_normsoftmax:
                for ti in range(len(self.n_class_list)):
                    loss += self.loss_fn[ti+4](y_hat[4], y[:, :, ti])   
                loss_div_cnt += 1
            
        loss /= loss_div_cnt
            
        return loss
    
    def save_checkpoint(self):
        saved_pt_list = glob(os.path.join(self.config.save_path, '*pth'))

        if len(saved_pt_list) > self.config.save_top_n:
            saved_pt_list = natsort.natsorted(saved_pt_list)

            for li in saved_pt_list[:-(self.config.save_top_n+1)]:
                os.remove(li)

        save_path = '{}/epoch:{}-{}:{:.4f}.pth'.format(
                    self.config.save_path,
                    self.current_epoch,
                    self.config.target_metric,
                    self.metric_helper.get_best_metric(),
                )

        if self.config.num_gpus > 1:
            ckpt_state = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.module.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.current_epoch,
            }
        else:
            ckpt_state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.current_epoch,
            }

        torch.save(ckpt_state, save_path)

        print('[+] save checkpoint : ', save_path)