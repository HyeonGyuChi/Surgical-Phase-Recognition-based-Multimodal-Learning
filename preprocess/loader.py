import os
from glob import glob
import natsort
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import pandas as pd

EXCEPTION_NUM = -1000000

class PETRAWBBOXLoader():
    def __init__(self, root_dir, dsize):
        self.root_dir = root_dir
        self.dsize = dsize

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

    def load_data(self, objs):
        
        bbox_data = {}

        frame_list = glob(self.root_dir + '/*')
        frame_list = natsort.natsorted(frame_list)

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

