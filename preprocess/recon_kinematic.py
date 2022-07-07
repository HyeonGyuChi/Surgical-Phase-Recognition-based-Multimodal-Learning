import os
from glob import glob
import natsort
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import pandas as pd
from itertools import combinations
from shutil import copyfile

from recon_kinematic_helper import get_bbox_loader, set_bbox_loader, get_bbox_obj_info, get_recon_method, normalized_pixel, denormalized_pixel, standardization

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

    def reconstruct(self, methods, extract_objs, extract_pairs, is_normalized=False, is_standardization=True):

        recon_df = pd.DataFrame([]) # save and return
        columns = [] # in recon_df
        entities_start_ids = {}

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
        entities_np = entities_np.astype(np.float64) # int => float for normalized

        print('entities - {}'.format(entities_np.shape))
            
        # append recon df
        if is_normalized: # normalized
            norm_entities_np = np.full_like(entities_np, fill_value=EXCEPTION_NUM)
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
                        kine_results = recon_method(target_np, interval_sec= self.fps)

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

        # standarization
        if is_standardization:
            print('\n[+] \t standardization ...')
            all_value = recon_df.values

            standardization_value = np.full(all_value.shape, fill_value=-2, dtype=np.float64)

            d_idx, feat_idx = all_value.shape
            print('all value :', all_value)
            print('all value shape:', all_value.shape)
            for feat_i in range(feat_idx): # only standardization with EXCEPTION
                recon_ids = np.where(all_value[:, feat_i] != EXCEPTION_NUM)
                
                mean, std = np.mean(all_value[recon_ids, feat_i]), np.std(all_value[recon_ids, feat_i])
                standardization_value[recon_ids, feat_i] = (all_value[recon_ids, feat_i] - mean) / (std + 1e-6)

            recon_df = pd.DataFrame(standardization_value, columns=recon_df.columns)
            print(recon_df.describe())
            print('\n[-] \t standardization ...')
        
        # ref: https://tariat.tistory.com/583
        recon_df.to_pickle(os.path.splitext(self.save_path)[0] + '.pkl')
        recon_df.to_csv(self.save_path)        
        
        return recon_df

if __name__ == "__main__":
    
    base_path = '/raid/multimodal'

    data_root_path = base_path + '/PETRAW/Training'
    target_root_path = data_root_path + '/Segmentation_swin'
    save_root_path = data_root_path + '/Kinematic_swin_new'

    file_list = natsort.natsorted(os.listdir(target_root_path))
    
    rk = recon_kinematic("", "", fps=5, sample_interval=300, dsize=(512, 512), task='PETRAW') # sample rate 6 (5fps)

    extract_objs = ['Grasper']
    extract_pairs = [('Grasper', 'Grasper')]

    # extract_objs = ['Background',
    #             'HarmonicAce_Head','HarmonicAce_Body','MarylandBipolarForceps_Head',
    #             'MarylandBipolarForceps_Wrist','MarylandBipolarForceps_Body',
    #             'CadiereForceps_Head','CadiereForceps_Wrist','CadiereForceps_Body',
    #             'CurvedAtraumaticGrasper_Head','CurvedAtraumaticGrasper_Body',
    #             'Stapler_Head','Stapler_Body',
    #             'Medium-LargeClipApplier_Head','Medium-LargeClipApplier_Wrist','Medium-LargeClipApplier_Body',
    #             'SmallClipApplier_Head','SmallClipApplier_Wrist','SmallClipApplier_Body',
    #             'Suction-Irrigation','Needle',
    #             'Endotip','Specimenbag','DrainTube','Liver','Stomach',
    #             'Pancreas','Spleen','Gallbbladder','Gauze','The_Other_Inst','The_Other_Tissue']

    # extract_pairs = combinations_of_objs = list(combinations(extract_objs, 2)) # total obj pair
    # for i, p in enumerate(extract_pairs):
    #     print('{}: {}'.format(i, p))

    methods = ['centroid', 'eoa', 'partial_pathlen', 'cumulate_pathlen', 'speed', 'velocity', 'IoU', 'gIoU', 'dIoU', 'cIoU']
    
    
    # for PETRAW, concat GASTRIC
    for key_val in file_list:
        target_path = target_root_path + '/{}'.format(key_val)
        save_path = save_root_path + '/{}_seg_ki.csv'.format(key_val)

        os.makedirs(save_root_path, exist_ok=True)
        rk.set_path(target_path, save_path)
        recon_df = rk.reconstruct(methods, extract_objs, extract_pairs)

    # for GASTRIC
    '''
    for key_val in file_list: # pateint
        video_list = natsort.natsorted(os.listdir(os.path.join(target_root_path, key_val)))
        for video_name in video_list:
            target_path = target_root_path + '/{}/{}'.format(key_val, video_name)
            save_path = save_root_path + '/{}/{}_seg_ki.csv'.format(key_val, video_name)
            
            os.makedirs(os.path.join(save_root_path, 'key_val'), exist_ok=True)
            rk.set_path(target_path, save_path)
            recon_df = rk.reconstruct(methods, extract_objs, extract_pairs)
    '''

    print('done')

def concat_gastric_seg_data():
    base_path = '/raid/multimodal'

    data_root_path = base_path + '/gastric'
    target_root_path = data_root_path + '/Segmentation_deeplabv3'
    save_root_path = data_root_path + '/Segmentation_deeplabv3_concat'

    file_list = natsort.natsorted(os.listdir(target_root_path))

    # concatnate
    for key_val in file_list: # pateint
        video_list = natsort.natsorted(os.listdir(os.path.join(target_root_path, key_val)))
        for video_name in video_list:
            target_path = target_root_path + '/{}/{}'.format(key_val, video_name)
            save_path = save_root_path + '/{}'.format(key_val)
            os.makedirs(save_path, exist_ok=True)

            f_list = glob(os.path.join(target_path, '*.gz'))

            for src in f_list:
                dst_filename = '{}-{}'.format(src.split('/')[-2], os.path.basename(src))
                dst = os.path.join(save_path, dst_filename)
                print('src: ', src)
                print('dst: ', dst)
                print('-----' * 4)
                copyfile(src, dst)
    
    print('done')