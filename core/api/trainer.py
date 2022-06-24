import os
import torch
import torch.nn as nn
import numpy as np
import json
import natsort
import time
from glob import glob
from tqdm import tqdm
from natsort import natsort

from core.model import get_model, get_fusion_model, get_loss, configure_optimizer
from core.dataset import get_dataset
from core.config.task_info import task_dict
from core.utils.metric import MetricHelper


class Trainer():
    # Multimodal Trainer
    def __init__(self, args: dict):
        self.args = args
        self.setup()
        
    def setup(self):
        self.current_epoch = 1
        
        # make log directory
        self.set_save_path()

        # save hyperparams
        param_dict = vars(self.args)
        with open(self.args.save_path + '/params.json', 'w') as f:
            json.dump(param_dict, f, indent=4)

        # Load dataset
        print('======= Load Dataset =======')
        self.train_loader, self.val_loader, ski_feature_num = get_dataset(self.args, return_ski_feature_num=True) # @HG modify
        self.set_class_weight()

        # TODO: lstm feature 개수변경 (for variaty of ski_method)
        print('SKI FEATURE NUM: ', ski_feature_num)
        self.args.input_size = ski_feature_num # @HG modify
        
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
        
        # Load loss function
        print('======= Load Loss Function =======')
        self.loss_fn = get_loss(self.args)
        
        # Load Optimizer, scheduler
        print('======= Load Optimizers =======')
        self.optimizer, self.scheduler = configure_optimizer(self.args, self.model)
    
        # Set multi-gpu
        if self.args.num_gpus > 1:
            print('======= Set Multi-GPU =======')
            self.model = nn.DataParallel(self.model, 
                                        device_ids=list(range(self.args.num_gpus))).cuda()
        else:
            self.model.to(self.args.device)
            
        # Load pre-trained model
        if self.args.restore_path is not None:
            print('======= Load Pretrained Model =======')
            
            states = torch.load(self.args.restore_path)

            if 'state_dict' in states:
                self.model.load_state_dict(states['state_dict'])
            else:     
                self.model.load_state_dict(states['model'])
                
                if self.args.resume:
                    self.optimizer.load_state_dict(states['optimizer'])
                    self.scheduler.load_state_dict(states['scheduler'])
                    self.current_epoch = states['epoch']
        
        print('======= Set Metric Helper =======')
        self.metric_helper = MetricHelper(self.args)
        
    def set_save_path(self):
        self.args.save_path += '/{}-{}_loss-{}-{}-'.format(
            self.args.model,
            self.args.loss_fn,
            self.args.dataset,
            self.args.task,
        )
        dtype_list = 'dtype['

        for i, dtype in enumerate(self.args.data_type):
            dtype_list += '{}'.format(dtype)
            if i+1 != len(self.args.data_type):
                dtype_list += '-'

        self.args.save_path += '{}]-1'.format(dtype_list)

        if os.path.exists(self.args.save_path):
            for idx in range(2, 1000):
                tmp_save_path = self.args.save_path[:-2] + '-{}'.format(idx)
                
                if not os.path.exists(tmp_save_path):
                    self.args.save_path = tmp_save_path
                    break

        os.makedirs(self.args.save_path, exist_ok=True)

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

        # task 별 class 수 설정
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

    def fit(self):
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.args.max_epoch + 1):
            self.current_epoch = epoch
            self.metric_helper.update_epoch(epoch)
            
            # train phase
            self.train()
            
            # validation phase
            self.valid()

        return self.args

    def train(self):
        self.model.train()
        
        for data in tqdm(self.train_loader, desc='[Epoch {} - Train Phase] : '.format(self.current_epoch)):
            self.optimizer.zero_grad()
            
            x, y = data
            if self.args.device == 'cuda':
                for k in x.keys():
                    x[k] = x[k].to(self.args.device)
                y = y.to(self.args.device)
                
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
            if self.args.device == 'cuda':
                for k in x.keys():
                    x[k] = x[k].to(self.args.device)
                y = y.to(self.args.device)
                
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
        
        if self.args.lr_scheduler == 'reduced':
            self.scheduler.step(self.metric_helper.get_loss('valid'))
        else:
            self.scheduler.step()
        
        if self.metric_helper.update_best_metric(metric): # save best metric
            self.save_checkpoint()

        if self.current_epoch == self.args.max_epoch: # last epoch checkpoint saving
            # get current target metric
            target_met = 0
            for met in metric:
                target_met += met[self.args.target_metric]
            
            target_met /= len(metric)

            self.save_last_checkpoint(target_met)
        
    def forward(self, x, y):
        if self.args.model == 'multi':
            y_hat, fuse_loss = self.model(x)
            
            loss = self.calc_loss(y_hat, y)
            
            if fuse_loss is not None:
                loss += torch.mean(fuse_loss)
        elif self.args.model == 'slowfast':
            y_hat = self.model.forward(x['video'], return_loss=True, infer_3d=True)

            loss = self.calc_loss(y_hat, y)
        else:
            y_hat = self.model(x)
        
            loss = self.calc_loss(y_hat, y)
        
        return y_hat, loss 
        
    def calc_loss(self, y_hat, y):
        # y_hat : N_task x (B x seq X C)
        # y : B x seq x (N_task classes)        
        loss = 0
        loss_div_cnt = 0

        if 'ce' in self.args.loss_fn:
            for ti in range(len(self.n_class_list)):
                # for seq in range(y.shape[1]):
                    # loss += self.loss_fn(y_hat[ti][:, seq, ], y[:, seq, ti])
                loss += self.loss_fn(y_hat[ti][:, ], y[:, 0, ti])
                loss_div_cnt += 1

        else: # cb, bs, eqlv2
            for ti in range(len(self.n_class_list)):
                # for seq in range(y.shape[1]):
                    # loss += self.loss_fn[ti](y_hat[ti][:, seq, :], y[:, seq, ti])
                loss += self.loss_fn[ti](y_hat[ti][:, :], y[:, 0, ti])
                loss_div_cnt += 1
                    
            if self.args.use_normsoftmax:
                for ti in range(len(self.n_class_list)):
                    loss += self.loss_fn[ti+4](y_hat[4], y[:, :, ti])   
                loss_div_cnt += 1
            
        # loss /= loss_div_cnt
            
        return loss
    
    def save_checkpoint(self):
        saved_pt_list = []

        for fname in os.listdir(self.args.save_path):
            if fname[-3:] == 'pth':
                saved_pt_list.append(self.args.save_path + '/{}'.format(fname))

        # top N개 빼고 삭제
        if len(saved_pt_list) > self.args.save_top_n:
            saved_pt_list = natsort.natsorted(saved_pt_list)

            for li in saved_pt_list[:-(self.args.save_top_n+1)]:
                os.remove(li)

        # 현재 값들 (checkpoints)
        if self.args.num_gpus > 1:
            ckpt_state = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
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

        save_path = '{}/epoch:{}-{}:{:.4f}.pth'.format(
                    self.args.save_path,
                    self.current_epoch,
                    self.args.target_metric,
                    self.metric_helper.get_best_metric(),
                )

        
        # best model 저장
        torch.save(ckpt_state, save_path)

        print('[+] save checkpoint : ', save_path)

    def save_last_checkpoint(self, last_metric):
        # last checkpoint saving, best epoch save
        
        # total best model 저장 => add -best 및 set restore_path
        pt_list = []
        dir_list = os.listdir(self.args.save_path)
        
        for pt in dir_list:
            if os.path.splitext(pt)[-1] == '.pth':
                pt_list.append(pt)
        
        best_pt_path = os.path.join(self.args.save_path, natsort.natsorted(pt_list)[-1])
        rename_pt_path = os.path.join(self.args.save_path, os.path.splitext(os.path.basename(best_pt_path))[0] + '-best' + '.pth')
        print('best:', best_pt_path)
        print('rename:', rename_pt_path)
        os.rename(best_pt_path, rename_pt_path)

        self.args.restore_path = rename_pt_path # set restore_path

        # 현재 값들 (checkpoints)
        if self.args.num_gpus > 1:
            ckpt_state = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
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

        save_path = '{}/epoch:{}-{}:{:.4f}-last.pth'.format(
                    self.args.save_path,
                    self.current_epoch,
                    self.args.target_metric,
                    last_metric,
                )

        torch.save(ckpt_state, save_path)
        print('[+] save checkpoint (Last Epoch) : ', save_path)