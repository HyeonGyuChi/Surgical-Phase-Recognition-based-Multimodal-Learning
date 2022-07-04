import os
from glob import glob
import natsort
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import pandas as pd
from itertools import combinations

from recon_kinematic_helper import get_bbox_loader, set_bbox_loader, get_bbox_obj_info, get_recon_method, normalized_pixel, denormalized_pixel

EXCEPTION_NUM = -999

class recon_kinematic():
    def __init__(self, target_path, save_path, fps, sample_interval, dsize=(512, 512), task='PETRAW'):
        self.target_path = target_path
        self.save_path = save_path

        # hyper config
        self.dsize = dsize # w, h
        self.sample_interval = sample_interval # sampling interval from segmentation imgs
        self.fps = fps # for calc interval sec in [velocity, speed]

        # bbox dataloader setup
        self.bbox_loader = get_bbox_loader(task, self.target_path, self.dsize, self.sample_interval)

    def set_path(self, target_path, save_path):
        self.target_path, self.save_path = target_path, save_path
        self.bbox_loader = set_bbox_loader(self.bbox_loader, self.target_path, self.dsize)

    def reconstruct(self, methods, extract_objs, extract_pairs, is_normalized=True):

        recon_df = pd.DataFrame([]) # save and return
        columns = [] # in recon_df
        entities_start_ids = {}
        
        # combinations_of_objs = list(combinations(extract_objs, 2)) # total obj pair

        print('\n[+] \t load bbox data ... {}'.format(extract_objs))
        bbox_data = self.bbox_loader.load_data(extract_objs)
    
        # save data
        '''
        with open('/dataset3/multimodal/PETRAW/Training/Seg_kine8/bbox_data.pickle','wb') as fw:
            pickle.dump(bbox_data, fw)

        # load data
        with open('/dataset3/multimodal/PETRAW/Training/Seg_kine8/bbox_data.pickle','rb') as fr:
            bbox_data = pickle.load(fr)
        '''

        obj_key, obj_to_color = get_bbox_obj_info(self.bbox_loader)
        print('[-] \t load bbox data ... {}'.format(extract_objs))

        print('\n[+] \t arrange entity index ... {}'.format(extract_objs))
        entities_cnt = 0
        entities_np = []
        entity_col = ['x_min', 'x_max', 'y_min', 'y_max'] # VOC style
        
        for obj in extract_objs: # (idx, bbox1 points + bbox2 points + ...)
            print('{} - {}'.format(obj, bbox_data[obj].shape))

            entity_cnt_in_obj = bbox_data[obj].shape[1] // len(entity_col)
            entities_np.append(bbox_data[obj])

            entitiy_ids = []
            for i in range(entity_cnt_in_obj):
                entitiy_ids.append(len(entity_col) * (entities_cnt + i))
                columns += ['{}_{}-{}'.format(obj, i, col) for col in entity_col]
            
            entities_start_ids[obj] = entitiy_ids
            entities_cnt += entity_cnt_in_obj

        # list to np
        entities_np = np.hstack(entities_np) # pixel
        print('entities - {}'.format(entities_np.shape))
            
        # append recon df
        if is_normalized: # normalized 
            norm_entities_np = np.zeros(entities_np.shape)
            w, h = self.dsize
            for k, wt in enumerate([w, w, h, h] * entities_cnt):
                for f_idx in range(entities_np.shape[0]):
                    if not entities_np[f_idx, k] == EXCEPTION_NUM:
                        norm_entities_np[f_idx, k] = normalized_pixel(entities_np[f_idx, k], wt)
            
            recon_df = pd.concat([recon_df, pd.DataFrame(norm_entities_np)], axis=1) # column bind
        
        else:
            recon_df = pd.concat([recon_df, pd.DataFrame(entities_np)], axis=1) # column bind

        recon_df.columns = columns

        print(entities_start_ids)
        
        print(recon_df)
        print('[-] \t arrange entity index ... {}'.format(extract_objs))

        # reconstruct
        print('\n[+] \t reconstruct ... {} => {}'.format(extract_objs, extract_pairs))

        print('\n[+] \t single reconstruct ... {}'.format(extract_objs))

        single_methods = [m for m in methods if m in ['centroid', 'eoa', 'partial_pathlen', 'cumulate_pathlen', 'speed', 'velocity']]
        pair_methods = [m for m in methods if m in ['IoU', 'gIoU', 'dIoU', 'cIoU']]
        
        print('single method: ', single_methods)

        # recon single value
        for method in single_methods: # ['centroid', 'eoa', 'pathlen', ..]
            for obj in extract_objs: # ['Grasper', 'Blocks', 'obj3', ..]
                for i, start_idx in enumerate(entities_start_ids[obj]):
                    kine_results = []
                    recon_method, recon_method_col, norm_weights = get_recon_method(method, self.dsize)
                    target_entity = '{}_{}'.format(obj, i)
                    
                    target_np = entities_np[:, start_idx: start_idx + len(entity_col)] # entitiy bbox info
                    
                    # calc from single rows : apply method from frame by frame
                    if method in ['centroid', 'eoa']:
                        for f_idx in range(target_np.shape[0]):
                            if method == 'eoa':
                                result = recon_method(target_np[f_idx, :], self.dsize)
                            else: 
                                result = recon_method(target_np[f_idx, :])
                                
                            kine_results.append(result)

                        kine_results = np.stack(kine_results) # list to np
                    
                    # calc from multiple rows : apply method from all frame
                    if method in ['cumulate_pathlen']:
                        kine_results = recon_method(target_np)
                    
                    if method in ['partial_pathlen']:
                        kine_results = recon_method(target_np, window_size=8)

                    if method in ['speed', 'velocity']:
                        kine_results = recon_method(target_np, interval_sec= 1 / self.fps * 8)

                    # normalized 
                    if is_normalized:
                        for k, wt in enumerate(norm_weights):
                            for f_idx in range(target_np.shape[0]):
                                if not kine_results[f_idx, k] == EXCEPTION_NUM:
                                    kine_results[f_idx, k] = normalized_pixel(kine_results[f_idx, k], wt)

                    # append recon df
                    recon_df = pd.concat([recon_df, pd.DataFrame(kine_results)], axis=1) # column bind
                    columns += ['{}-{}'.format(target_entity, col) for col in recon_method_col]
                    recon_df.columns = columns

        print(recon_df.shape)
        print(recon_df.columns)
        # print(recon_df.iloc[:, 20:24]) # centroid
        # print(recon_df.iloc[:, 30:32]) # eoa

        print('\n[-] \t single reconstruct ... {}'.format(extract_objs))

        print('\n[+] \t pair reconstruct ... {}'.format(extract_pairs))
        print('pair methods: ', pair_methods)

        # recon pair value
        for method in pair_methods: # ['IoU', 'gIoU', 'dIoU', 'cIoU']
            for src_obj, target_obj in extract_pairs: # ('Grasper', 'Grasper'), ('Grasper', 'Blocks') .. 
                for i, src_start_idx in enumerate(entities_start_ids[src_obj]): # src per entiity
                    if src_obj == target_obj and i > 0 : break  # same obj, calc only one time

                    for j, target_start_idx in enumerate(entities_start_ids[target_obj]): # target per entiity
                        if src_obj == target_obj and i == j: # don't calc with same entitiy
                            continue

                        kine_results = []
                        recon_method, recon_method_col, norm_weights = get_recon_method(method, self.dsize)
                        
                        src_entity = '{}_{}'.format(src_obj, i)
                        target_entity = '{}_{}'.format(target_obj, j)
                        
                        src_np = entities_np[:, src_start_idx: src_start_idx + len(entity_col)] # entitiy bbox info
                        target_np = entities_np[:, target_start_idx: target_start_idx + len(entity_col)] # entitiy bbox info
                    
                        # calc from single rows : apply method from frame by frame
                        if method in ['IoU', 'gIoU', 'dIoU', 'cIoU']:
                            for f_idx in range(src_np.shape[0]):
                                result = recon_method(src_np[f_idx, :], target_np[f_idx, :]) # pair numpy input
                                kine_results.append(result)
                                
                            kine_results = np.stack(kine_results) # list to np

                        # normalized 
                        if is_normalized:
                            for k, wt in enumerate(norm_weights):
                                for f_idx in range(src_np.shape[0]):
                                    if not kine_results[f_idx, k] == EXCEPTION_NUM:
                                        kine_results[f_idx, k] = normalized_pixel(kine_results[f_idx, k], wt)

                        # append recon df
                        recon_df = pd.concat([recon_df, pd.DataFrame(kine_results)], axis=1) # column bind
                        columns += ['{}-{}-{}'.format(src_entity, target_entity, col) for col in recon_method_col]
                        recon_df.columns = columns


        # TODO - save recon df
        print(recon_df.shape)
        print(recon_df.columns)

        print('\n[-] \t pair reconstruct ... {}'.format(extract_pairs))
        
        print('\n[-] \t reconstruct ... {} => {}'.format(extract_objs, extract_pairs))

        # ref: https://tariat.tistory.com/583
        recon_df.to_pickle(os.path.splitext(self.save_path)[0] + '.pkl')
        recon_df.to_csv(self.save_path)        
        
        return recon_df

if __name__ == "__main__":

    base_path = '/raid/multimodal'

    data_root_path = base_path + '/PETRAW/Training'
    target_root_path = data_root_path + '/Segmentation'
    save_root_path = data_root_path + '/Kinematic_segmentation'

    file_list = natsort.natsorted(os.listdir(target_root_path))
    
    rk = recon_kinematic("", "", fps=30, sample_interval=6) # sample rate 6 (5fps)

    # extract_objs = ['Grasper', 'Blocks']
    # extract_pairs = [('Grasper', 'Grasper'), ('Grasper', 'Blocks')]

    extract_objs = ['Grasper']
    extract_pairs = [('Grasper', 'Grasper')]

    methods = ['centroid', 'eoa', 'partial_pathlen', 'cumulate_pathlen', 'speed', 'velocity', 'IoU', 'gIoU', 'dIoU', 'cIoU']
    
    for key_val in file_list:
        target_path = target_root_path + '/{}'.format(key_val)
        # save_path = save_root_path + '/{}_seg_ki.pkl'.format(key_val)
        save_path = save_root_path + '/{}_seg_ki.csv'.format(key_val)

        os.makedirs(save_root_path, exist_ok=True)
        rk.set_path(target_path, save_path)
        recon_df = rk.reconstruct(methods, extract_objs, extract_pairs)
    