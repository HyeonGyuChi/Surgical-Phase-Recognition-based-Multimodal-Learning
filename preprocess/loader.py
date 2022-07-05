import os
from glob import glob
import natsort
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import pandas as pd
import gzip

EXCEPTION_NUM = -999

class PETRAWBBOXLoader():
    def __init__(self, root_dir, dsize, sample_interval):
        self.root_dir = root_dir
        self.dsize = dsize
        self.sample_interval = sample_interval # set 6, (30fps -> 5fps)

        self.obj_key = ['Grasper', 'Blocks', 'obj3', 'obj4', 'obj5']

        self.obj_to_color = {
                'Grasper': [4,2],
                'Blocks': [5],
                'obj3': [7,9],
                'obj4': [11,13],
                'obj5': [14,32,64],
            }

    def set_root_dir(self, root_dir): # should set
        self.root_dir = root_dir
    
    def set_dsize(self, dsize): # should set
        self.dsize = dsize # w, h

    def set_sample_interval(self, sample_interval): # should set
        self.sample_interval = sample_interval  # set 6, (30fps -> 5fps)

    def load_data(self, objs):
        
        bbox_data = {}

        frame_list = glob(self.root_dir + '/*')
        frame_list = natsort.natsorted(frame_list)[::self.sample_interval]

        for fi, fpath in enumerate(tqdm(frame_list)):
            # extract bbox each tools
            if '.npz' == os.path.splitext(self.root_dir)[1]:
                img = np.load(fpath)['arr_0']
            else:
                img = cv2.imread(fpath)
                img = cv2.resize(img, dsize=self.dsize) # 1080x1920 => 512x512

            h,w,c = img.shape

            new_img = np.zeros((h,w)) # seg info

            ids = np.where(img[:,:,0]>0) # blue // 001
            if len(ids[0]) > 0:
                new_img[ids[0], ids[1]] = 1
            
            ids = np.where(img[:,:,1]>0) # green // 010
            if len(ids[0]) > 0:
                new_img[ids[0], ids[1]] += 2
            
            ids = np.where(img[:,:,2]>0) # red // 100
            if len(ids[0]) > 0:
                new_img[ids[0], ids[1]] += 4

            objs_bbox = self._get_bbox(new_img, objs)

            for obj in objs: # ['Grasper', 'Blocks', ..]
                if obj in bbox_data:
                    bbox_data[obj].append(objs_bbox[obj].ravel())
                else:
                    bbox_data[obj] = []
                    bbox_data[obj].append(objs_bbox[obj].ravel())

        # list to numpy
        for obj in objs: # ['Grasper', 'Blocks', ..]
            bbox_data[obj] = np.stack(bbox_data[obj])
        
        return bbox_data
    
    '''
    def _bbox_to_normalized_pixel(self, bbox, image_size):
        x_min, x_max, y_min, y_max = bbox[:] # VOC style
        img_w, img_h = image_size
 
        bbox[:2] = normalized_pixel(bbox[:2], img_w) # x min max
        bbox[2:] = normalized_pixel(bbox[2:], img_h) # y
        
        return bbox
    '''

    def _get_bbox(self, new_img, objs):
        """
        return
        obj_bbox = {
            'Grasper': [[x min, x max, y min, y max], [x min, x max, y min, y max]]
            'Blocks': [[x_m, .. ]]
            ...
        } // return bbox each tools on left, right hands
        """
        objs_bbox = {}

        for obj in objs: # ['Grasper', 'Blocks' ,..]
            objs_bbox[obj] = []

            for _ in range(len(self.obj_to_color[obj])): # initialize l,r
                objs_bbox[obj].append([])

            for i, color in enumerate(self.obj_to_color[obj]): # each entitiy 
                ids = np.where(new_img == color)
                
                if len(ids[0]) > 0: # exist // ids[0] - h, ids[1] - w
                    x_min, x_max, y_min, y_max = np.min(ids[1]), np.max(ids[1]), np.min(ids[0]), np.max(ids[0]) # VOC style, cf. COCO/YOLO style [x min, y max, width, height]
                    bbox = (x_min, x_max, y_min, y_max)
                    # bbox = self._bbox_to_normalized_pixel(bbox, self.dsize) # norm [x min, x max, y min, y max]

                else : # non
                    bbox = [EXCEPTION_NUM, EXCEPTION_NUM, EXCEPTION_NUM, EXCEPTION_NUM]

                objs_bbox[obj][i] = bbox
            
            # convert list to numpy
            objs_bbox[obj] = np.array(objs_bbox[obj])
        
        return objs_bbox


class GASTRICBBOXLoader():
    def __init__(self, root_dir, dsize, sample_interval):
        self.root_dir = root_dir
        self.dsize = dsize
        self.sample_interval = sample_interval # set 6, (30fps -> 5fps)

        self.obj_key = ['Background',
                'HarmonicAce_Head','HarmonicAce_Body','MarylandBipolarForceps_Head',
                'MarylandBipolarForceps_Wrist','MarylandBipolarForceps_Body',
                'CadiereForceps_Head','CadiereForceps_Wrist','CadiereForceps_Body',
                'CurvedAtraumaticGrasper_Head','CurvedAtraumaticGrasper_Body',
                'Stapler_Head','Stapler_Body',
                'Medium-LargeClipApplier_Head','Medium-LargeClipApplier_Wrist','Medium-LargeClipApplier_Body',
                'SmallClipApplier_Head','SmallClipApplier_Wrist','SmallClipApplier_Body',
                'Suction-Irrigation','Needle',
                'Endotip','Specimenbag','DrainTube','Liver','Stomach',
                'Pancreas','Spleen','Gallbbladder','Gauze','The_Other_Inst','The_Other_Tissue']
        
        # color map
        self.obj_to_color = { # channel number
            'Background': [1], 
            'HarmonicAce_Head': [2],
            'HarmonicAce_Body': [3],
            'MarylandBipolarForceps_Head': [4],
            'MarylandBipolarForceps_Wrist': [5],
            'MarylandBipolarForceps_Body': [6],
            'CadiereForceps_Head': [7],
            'CadiereForceps_Wrist': [8],
            'CadiereForceps_Body': [9],
            'CurvedAtraumaticGrasper_Head': [10],
            'CurvedAtraumaticGrasper_Body': [11],
            'Stapler_Head': [12],
            'Stapler_Body': [13],
            'Medium-LargeClipApplier_Head': [14],
            'Medium-LargeClipApplier_Wrist': [15],
            'Medium-LargeClipApplier_Body': [16],
            'SmallClipApplier_Head': [17],
            'SmallClipApplier_Wrist': [18],
            'SmallClipApplier_Body': [19],
            'Suction-Irrigation': [20],
            'Needle': [21],
            'Endotip': [22],
            'Specimenbag': [23],
            'DrainTube': [24],
            'Liver': [25],
            'Stomach': [26],
            'Pancreas': [27],
            'Spleen': [28],
            'Gallbbladder': [29],
            'Gauze': [30],
            'The_Other_Inst': [31],
            'The_Other_Tissue': [32],
            }
        
    def set_root_dir(self, root_dir): # should set
        self.root_dir = root_dir
    
    def set_dsize(self, dsize): # should set
        self.dsize = dsize # w, h

    def set_sample_interval(self, sample_interval): # should set
        self.sample_interval = sample_interval  # set 6, (30fps -> 5fps)

    def load_data(self, objs):
        
        bbox_data = {}

        frame_list = glob(self.root_dir + '/*')
        frame_list = natsort.natsorted(frame_list)[::self.sample_interval]

        for fi, fpath in enumerate(tqdm(frame_list)):

            # extract bbox each tools
            with gzip.GzipFile(fpath, "r") as f: # .gz
                img = np.load(f, allow_pickle=True) # 512x512
    
            h,w,c = img.shape

            new_img = np.zeros((h,w)) # seg info

            # to new img(1 channel) from 32 channel
            for c_idx in range(1, c): # 32 channel
                ids = np.where(img[:,:,c_idx] > 0)

                if len(ids[0]) > 0:
                    new_img[ids[0], ids[1]] = c_idx

            objs_bbox = self._get_bbox(new_img, objs)

            for obj in objs: # ['HarmonicAce_Head', 'HarmonicAce_Body' ,..]
                if obj in bbox_data:
                    bbox_data[obj].append(objs_bbox[obj].ravel())
                else:
                    bbox_data[obj] = []
                    bbox_data[obj].append(objs_bbox[obj].ravel())

        # list to numpy
        for obj in objs: # ['HarmonicAce_Head', 'HarmonicAce_Body' ,..]
            bbox_data[obj] = np.stack(bbox_data[obj])
        
        return bbox_data

    def _get_bbox(self, new_img, objs):
        """
        return
        obj_bbox = {
            'Grasper': [[x min, x max, y min, y max], [x min, x max, y min, y max]]
            'Blocks': [[x_m, .. ]]
            ...
        } // return bbox each tools on left, right hands
        """
        objs_bbox = {}

        for obj in objs: # ['HarmonicAce_Head', 'HarmonicAce_Body' ,..]
            objs_bbox[obj] = []

            for _ in range(len(self.obj_to_color[obj])): # initialize l,r
                objs_bbox[obj].append([])

            for i, color in enumerate(self.obj_to_color[obj]): # each entitiy 
                ids = np.where(new_img == color)
                
                if len(ids[0]) > 0: # exist // ids[0] - h, ids[1] - w
                    x_min, x_max, y_min, y_max = np.min(ids[1]), np.max(ids[1]), np.min(ids[0]), np.max(ids[0]) # VOC style, cf. COCO/YOLO style [x min, y max, width, height]
                    bbox = (x_min, x_max, y_min, y_max)
                    # bbox = self._bbox_to_normalized_pixel(bbox, self.dsize) # norm [x min, x max, y min, y max]

                else : # non
                    bbox = [EXCEPTION_NUM, EXCEPTION_NUM, EXCEPTION_NUM, EXCEPTION_NUM]

                objs_bbox[obj][i] = bbox
            
            # convert list to numpy
            objs_bbox[obj] = np.array(objs_bbox[obj])

        return objs_bbox