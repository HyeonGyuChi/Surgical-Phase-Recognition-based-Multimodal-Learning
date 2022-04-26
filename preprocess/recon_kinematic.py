import os
from glob import glob
import natsort
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import pandas as pd
from itertools import combinations

from recon_kinematic_helper import get_bbox_loader, set_bbox_loader, get_bbox_obj_info, get_recon_method

class recon_kinematic():
    def __init__(self, target_path, save_path, dsize=(512, 512), task='PETRAW'):
        self.target_path = target_path
        self.save_path = save_path

        # hyper config
        self.EXCEPTION_NUM = -100
        self.dsize = dsize # w, h

        # bbox dataloader setup
        self.bbox_loader = get_bbox_loader(task, self.target_path, self.dsize)

    def set_path(self, target_path, save_path):
        self.target_path, self.save_path = target_path, save_path
        self.bbox_loader = set_bbox_loader(self.bbox_loader, self.target_path, self.dsize)

    def reconstruct(self, methods, extract_objs, extract_pairs):

        recon_df = pd.DataFrame([]) # save and return
        columns = [] # in recon_df
        entities_start_ids = {}
        
        combinations_of_objs = list(combinations(extract_objs, 2)) # total obj pair

        print('\n[+] \t load bbox data ... {}'.format(extract_objs))
        bbox_data = self.bbox_loader.load_data(extract_objs)
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
        entities_np = np.hstack(entities_np)
        print('entities - {}'.format(entities_np.shape))
        print(entities_start_ids)

        # append recon df
        recon_df = pd.concat([recon_df, pd.DataFrame(entities_np)], axis=1) # column bind
        recon_df.columns = columns
        print(recon_df)
        print('[-] \t arrange entity index ... {}'.format(extract_objs))

        # reconstruct
        print('\n[+] \t reconstruct ... {} => {}'.format(extract_objs, extract_pairs))

        single_methods = [m for m in methods if m in ['centroid', 'eoa', 'speed']]
        pair_methods = [m for m in methods if m in ['IoU', 'gIoU']]
        
        # recon single value
        for method in single_methods: # ['centroid', 'eoa', 'speed', ..]
            for obj in extract_objs: # ['Grasper', 'Blocks', 'obj3', ..]
                for i, start_idx in enumerate(entities_start_ids[obj]):
                    kine_results = []
                    recon_method, recon_method_col = get_recon_method(method)
                    target_entity = '{}_{}'.format(obj, i)
                    
                    target_np = entities_np[:, start_idx: start_idx + len(entity_col)] # entitiy bbox info
                    
                    # apply method frame by frame
                    for f_idx in range(target_np.shape[0]):
                        result = recon_method(target_np[f_idx, :])
                        kine_results.append(result)

                    kine_results = np.stack(kine_results) # list to np

                    # append recon df
                    recon_df = pd.concat([recon_df, pd.DataFrame(kine_results)], axis=1) # column bind
                    columns += ['{}-{}'.format(target_entity, col) for col in recon_method_col]
                    recon_df.columns = columns

            
        print(recon_df.shape)
        print(recon_df.iloc[:, 20:24]) # centroid
        print(recon_df.iloc[:, 30:32]) # eoa

        exit(0)
        # TODO - recon pair value
        # TODO - save recon df

        return recon_df

if __name__ == "__main__":

    base_path = '/dataset3/multimodal'

    data_root_path = base_path + '/PETRAW/Training'
    target_root_path = data_root_path + '/Segmentation'
    save_root_path = data_root_path + '/Seg_kine8'

    file_list = natsort.natsorted(os.listdir(target_root_path))
    
    rk = recon_kinematic("", "")

    methods = ['centroid', 'eoa']

    extract_objs = ['Grasper', 'Blocks', 'obj3']
    extract_pairs = ('Grasper', 'Blocks')
    
    for key_val in file_list:
        target_path = target_root_path + '/{}'.format(key_val)
        save_path = save_root_path + '/{}_seg_ki.pkl'.format(key_val)
        rk.set_path(target_path, save_path)
        rk.reconstruct(methods, extract_objs, extract_pairs)

        exit(0)
    