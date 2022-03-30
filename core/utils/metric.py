import torch
import math
import numpy as np
import os
import pandas as pd
from pycm import *
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from core.config.task_info import task_dict


class MetricHelper():
    """
        Help metric computation.
    """
    def __init__(self, config):
        self.config = config        
        self.n_classes = self.config.n_classes
        self.classes = list(range(self.n_classes))
        self.target_metric = self.config.target_metric
        if 'loss' in self.target_metric:
            self.best_metric = math.inf
        else:
            self.best_metric = 0
        self.epoch = 0
        
        self.loss_dict = {
            'train': [],
            'valid': [],
        }
        self.tmp_loss_dict = {
            'train': [],
            'valid': [],
        }
        
        n_result = task_dict[self.config.dataset][self.config.task][0]
        n_result2 = task_dict[self.config.dataset][self.config.task][-1]
        
        self.output_dict = {
            'pred': [list() for _ in range(n_result)],
            'gt': [list() for _ in range(n_result)],
        }
        
        self.results = np.zeros((self.config.max_epoch, n_result))
        self.results2 = np.zeros((self.config.max_epoch, n_result2))
        
    def update_epoch(self, epoch):
        self.epoch = epoch
        
    def write_preds(self, pred_list, gt_list):
        for idx in range(len(self.output_dict['pred'])):
            for pred, gt in zip(pred_list[idx], gt_list[idx]):
                self.output_dict['pred'][idx].append(pred.item())
                self.output_dict['gt'][idx].append(gt.item())
        
    def write_loss(self, loss_val, state='train'):
        self.tmp_loss_dict[state].append(loss_val)
            
    def update_loss(self, state='train'):
        avg_loss = sum(self.tmp_loss_dict[state]) / len(self.tmp_loss_dict[state])
        self.tmp_loss_dict[state] = []
        self.loss_dict[state].append(avg_loss)
            
    def calc_metric(self):
        """
            task metric computation
        """
        metric_list = []
        cnt = 0
        
        for idx in range(len(self.output_dict['pred'])):
            gt_list, pred_list = self.output_dict['gt'][idx], self.output_dict['pred'][idx]

            cm = ConfusionMatrix(gt_list, pred_list)
            bal_acc = balanced_accuracy_score(gt_list, pred_list)
            auc_list = list(cm.AUC.values())

            auroc_mean = 0
            for auc in auc_list:
                if auc is 'None':
                    auroc_mean += 0
                else:
                    auroc_mean += auc

            auroc_mean = auroc_mean / len(auc_list)
            
            gt_cnt_list = [len(np.where(np.array(gt_list) == i)[0]) for i in self.classes]
            pred_TP_list = [cm.TP[cls_name] for cls_name in cm.classes]
            p_acc = [p_TP / gt_cnt for p_TP, gt_cnt in zip(pred_TP_list, gt_cnt_list)]

            metrics = {
                'Epoch': self.epoch,
                'TP': [cm.TP[cls_name] for cls_name in cm.classes],
                'TN': [cm.TN[cls_name] for cls_name in cm.classes],
                'FP': [cm.FP[cls_name] for cls_name in cm.classes],
                'FN': [cm.FN[cls_name] for cls_name in cm.classes],
                'Accuracy': [cm.ACC[cls_name] for cls_name in cm.classes],
                'Precision': [cm.PPV[cls_name] for cls_name in cm.classes],
                'Recall': [cm.TPR[cls_name] for cls_name in cm.classes],
                'F1-Score': [cm.F1[cls_name] for cls_name in cm.classes],
                'Jaccard': [cm.J[cls_name] for cls_name in cm.classes],
                'Balance-Acc': bal_acc,
                'Percentage-Acc': p_acc,
                'AUROC': auroc_mean,
                'val_loss': self.loss_dict['valid'][-1],
            }
            
            # exception
            for k, v in metrics.items():
                if isinstance(v, list):
                    for vi, v2 in enumerate(v):
                        if v2 == 'None': # ConfusionMetrix return
                            metrics[k][vi] = 0
                        elif np.isinf(v2) or np.isnan(v2): # numpy return
                            metrics[k][vi] = 0
                else:
                    if v == 'None': # ConfusionMetrix return
                        metrics[k] = 0
                    elif np.isinf(v) or np.isnan(v): # numpy return
                        metrics[k] = 0
            
            self.output_dict['gt'][idx] = []
            self.output_dict['pred'][idx] = []
            
            metric_list.append(metrics)
            
            self.results[self.epoch-1, idx] = metrics['Balance-Acc']
            
            for ai, acc in enumerate(metrics['Percentage-Acc']):
                self.results2[self.epoch-1, cnt] = acc
                cnt += 1
        
            print('CLS BACC : {:.4f}'.format(metrics['Balance-Acc']))
        
        # save accuracy
        self.save_results()

        return metric_list
    
    def get_best_metric(self):
        return self.best_metric
            
    def get_loss(self, state='train'):
        return self.loss_dict[state][-1]
            
    def save_loss_pic(self):
        fig = plt.figure(figsize=(32, 16))

        plt.ylabel('Loss', fontsize=50)
        plt.xlabel('Epoch', fontsize=50)
        
        plt.plot(range(self.epoch), self.loss_dict['train'])
        plt.plot(range(self.epoch), self.loss_dict['valid'])
        
        plt.legend(['Train', 'Val'], fontsize=40)
        plt.savefig(self.config.save_path + '/loss.png')
        
        
    def update_best_metric(self, metric):
        target_met = 0    
        
        for met in metric:
            target_met += met[self.target_metric]
            
        target_met /= len(metric)
        
        if 'loss' in self.target_metric:    
            if self.best_metric > target_met:
                self.best_metric = target_met
                return True
        else:
            if self.best_metric < target_met:
                self.best_metric = target_met
                return True
        
        return False
        
    def save_results(self):
        cols = ['Balance-Acc' + f"_{i}" for i in range(self.results.shape[-1])] + ['Acc' + f"_{i}" for i in range(self.results2.shape[-1])]
        save_path = self.config.save_path + '/result_{}.csv'.format(self.config.model)
        
        data = [*list(self.results[self.epoch-1, :]), *list(self.results2[self.epoch-1, :])]

        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            print('Existed file loaded')
            
            new_df = pd.Series(data, index=cols)
            df = df.append(new_df, ignore_index=True)
            print('New line added')
            
        else:
            print('New file generated!')
            df = pd.DataFrame([data],
                        columns=cols
                        ) 

        df.to_csv(save_path, 
                index=False,
                float_format='%.4f')