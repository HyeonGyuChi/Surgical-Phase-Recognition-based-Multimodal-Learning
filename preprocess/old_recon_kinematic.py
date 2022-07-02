import os
from glob import glob
import natsort
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import pandas as pd

from preprocess.apply_recon_kinematic import get_visual_kinematic
from preprocess.load_data import load_patraw_segmap_data

class recon_kinematic():
    def __init__(self, target_path, save_path):
        self.target_path = target_path
        self.save_path = save_path
        
        # hyper config
        self.EXCEPTION_NUM = -100
        self.dsize = (512, 512)

    def set_path(self, target_path, save_path):
        self.target_path, self.save_path = target_path, save_path
    
    def _load_patraw_segmap_data(self, segmap_fpath): # segmap =(color bit mask)=> segdata
            
        seg_data = None

        if '.npz' == os.path.splitext(segmap_fpath)[1]:
            img = np.load(segmap_fpath)['arr_0']
        else:
            img = cv2.imread(segmap_fpath)
            img = cv2.resize(img, dsize=self.dsize) # 1080x1920 => 512x512

        h,w,c = img.shape
        seg_data = np.zeros((h,w)) # seg info
        
        ids = np.where(img[:,:,0]>0) # blue // 001
        if len(ids[0]) > 0:
            seg_data[ids[0], ids[1]] = 1
        
        ids = np.where(img[:,:,1]>0) # green // 010
        if len(ids[0]) > 0:
            seg_data[ids[0], ids[1]] += 2
        
        ids = np.where(img[:,:,2]>0) # red // 100
        if len(ids[0]) > 0:
            seg_data[ids[0], ids[1]] += 4

        
        return seg_data


    def _load_segmentation_data(self, tools=['Grasper']):
        
        bbox_data = {}
        
        frame_list = glob(self.target_path + '/*')
        frame_list = natsort.natsorted(frame_list)[:100]

        for fi, fpath in enumerate(tqdm(frame_list)):

            seg_data = self._load_patraw_segmap_data(fpath)
            
            # extract bbox each tools
            tools_bbox = self._get_bbox(new_img, tools)
            for tool in tools: # ['Grasper', 'block', ..]
                if tool in bbox_data:
                    bbox_data[tool].append(tools_bbox[tool].ravel())
                else:
                    bbox_data[tool] = []
                    bbox_data[tool].append(tools_bbox[tool].ravel())
        
        # stacking (list => np)
        seg_data = np.stack(seg_data) # (idx, h, w)
        
        # (idx, [l-x min, l-x max, l-y min, l-y max, r-x min, r-x max, r-y min, r-y max])
        for tool in tools:
            bbox_data[tool] = np.stack(bbox_data[tool])

        return seg_data, bbox_data

    def _get_bbox(self, new_img, tools):


    def reconstruct(self, methods, extract_tools, extract_pair):

        recon_df = pd.DataFrame([])

        extract_tools = ['Grasper', 'Blocks']

        print('[+] \t load segmentation ... {}'.format(key_val))
        seg_data, bbox_data = self._load_segmentation_data(extract_tools)
        print('[-] \t load segmentation ... {}'.format(key_val))

        print(seg_data.shape)
        print(bbox_data['Grasper'].shape)
        print(bbox_data['Blocks'].shape)

        # reconstruct
        print('[+] \t reconstruct ... {}'.format(key_val))

        for tool in extract_tools[:1]: # ['Grasper', 'Blocks', ...]
            bbox_df = pd.DataFrame(bbox_data[tool])
            recon_df = pd.concat([recon_df, bbox_df], axis=1) # column bind

            entity_cnt = bbox_df.shape[1] // len(col_name)
            
            base_col_name = ['x_min', 'x_max', 'y_min', 'y_max']
            col_name = []
            
            for i in range(entity_cnt):
                 # ['Blocks-0-x_min', 'Blocks-0-x_max', 'Blocks-0-y_min', 'Blocks-0-x_max']
                col_name += ['{}-{}-{}'.format(tool, i, col) in col for base_col_name]
            
            recon_df.columns = col_name

            for m in methods: # [centoird, eoa, IoU, ...]
                if m in ['centroid', 'eoa', 'speed']: # apply on one of entitiy
                    for i in range(entity_cnt):
                        target_np1 = bbox_data[:, len(col_name) * i : len(col_name) * (i + 1)]
                        kine_df = get_bbox_visual_kinematic(method=m, bbox_np1=target_np1)

                        col_name = ['{}-{}-{}'.format(tool, i, col) in col for col_name] # ['Blocks-0-x_min', 'Blocks-0-x_max', 'Blocks-0-y_min', 'Blocks-0-x_max']
                    
                elif method in ['IoU', 'gIoU']: # apply on two entitiy
                    target_np1, target_np2 = bbox_data[tool][:, 0:4], bbox_data[tool][:, 4:8]
                    kine_df = get_bbox_visual_kinematic(method=m, bbox_np1=target_np1, bbox_np2=target_np2)
                        
            
                if n_iter == 1:
                    target_np1, target_np2 = bbox_data[tool][:, 0:4], None
                    col_name = ['{}-M-{}'.format(tool, col) in col for col_name]
                
                elif n_iter == 2: # l,r
                    target_np1, target_np2 = bbox_data[tool][:, 0:4], bbox_data[tool][:, 4:8]
                    
                    col_name = ['{}-L-{}'.format(tool, col) in col for col_name] # ['Grasper-L-x_min', 'Grasper-L-x_max', 'Grasper-L-y_min', 'Grasper-L-x_max']
                    col_name += ['{}-R-{}'.format(tool, col) in col for col_name] # ['Grasper-R-x_min', 'Grasper-R-x_max', 'Grasper-R-y_min', 'Grasper-R-x_max']

                    if m in ['centroid', 'eoa', 'speed']:
                        kine_df = get_bbox_visual_kinematic(method=m, bbox_np1=target_np1, bbox_np2=target_np2)
                        

                recon_df.columns = col_name

                kine_df = get_bbox_visual_kinematic(method=m, bbox_np1=target_np1, bbox_np2=target_np2)
                recon_df = pd.concat([recon_df, kine_df], axis=1) # column bind

        print(recon_df)
        
        print('[-] \t reconstruct ... {}'.format(key_val))
    
        exit(0)

if __name__ == "__main__":

    base_path = '/dataset3/multimodal'

    data_root_path = base_path + '/PETRAW/Training'
    target_root_path = data_root_path + '/Segmentation'
    save_root_path = data_root_path + '/Seg_kine8'

    file_list = natsort.natsorted(os.listdir(target_root_path))

    rk = recon_kinematic("", "")

    methods = ['centroid', 'speed', 'eoa', 'giou']
    methods = ['centroid']
    
    for key_val in file_list:
        target_path = target_root_path + '/{}'.format(key_val)
        save_path = save_root_path + '/{}_seg_ki.pkl'.format(key_val)
        rk.set_path(target_path, save_path)

        rk.reconstruct(methods)

        exit(0)
    