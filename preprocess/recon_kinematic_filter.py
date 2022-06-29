# load exact visual kinematic featrue from recon kinematic set
from preprocess.recon_kinematic_helper import get_bbox_loader, set_bbox_loader, get_bbox_obj_info, get_recon_method
import pandas as pd
import pickle

class recon_kinematic_filter():
    def __init__(self, task='PETRAW'):
        self.task = task

    def set_src_path(self, src_path):
        self.src_path = src_path

    def filtering(self, methods, extract_objs, extract_pairs):
        columns = [] # parse columns from recon_df

        obj_key, obj_to_color = get_bbox_obj_info(get_bbox_loader(self.task, target_path='', dsize='', sample_rate='')) # dummy (target_path, dsize, sample_rate)
        
        # print('\n[+] \tsetting filter columns ... \n\tmethod : {} \n\textract_objs : {} ==> {}\n'.format(methods, extract_objs, extract_pairs))
        # base
        entity_col = ['x_min', 'x_max', 'y_min', 'y_max'] # VOC style

        for obj in extract_objs: # (idx, bbox1 points + bbox2 points + ...)
            for i in range(len(obj_to_color[obj])):
                columns += ['{}_{}-{}'.format(obj, i, col) for col in entity_col]

        single_methods = [m for m in methods if m in ['centroid', 'eoa', 'partial_pathlen', 'cumulate_pathlen', 'speed', 'velocity']]
        pair_methods = [m for m in methods if m in ['IoU', 'gIoU']]

        # single method
        for method in single_methods: # ['centroid', 'eoa', 'pathlen', ..]
            _, recon_method_col, _ = get_recon_method(method, img_size=(0,0)) # dummy (img_size)

            for obj in extract_objs: # ['Grasper', 'Blocks', 'obj3', ..]
                for i in range(len(obj_to_color[obj])):
                    target_entity = '{}_{}'.format(obj, i)
                    columns += ['{}-{}'.format(target_entity, col) for col in recon_method_col]

        # pair method
        for method in pair_methods: # ['IoU', 'gIoU']
            _, recon_method_col, _ = get_recon_method(method, img_size=(0,0)) # dummy (img_size)

            for src_obj, target_obj in extract_pairs: # ('Grasper', 'Grasper'), ('Grasper', 'Blocks') .. 
                for i in range(len(obj_to_color[src_obj])): # src per entiity
                    if src_obj == target_obj and i > 0 : break  # same obj, calc only one time

                    for j in range(len(obj_to_color[target_obj])): # target per entiity
                        if src_obj == target_obj and i == j: # don't calc with same entitiy
                            continue

                        src_entity = '{}_{}'.format(src_obj, i)
                        target_entity = '{}_{}'.format(target_obj, j)
                        columns += ['{}-{}-{}'.format(src_entity, target_entity, col) for col in recon_method_col]

        # print('NUM OF FEATURE : {}'.format(len(columns)))
        
        # print(columns)
        # print('\n[-] \tsetting filter columns ...')

        # print('\n[+] \tparsing from {} ...'.format(self.src_path))
        
        with open(self.src_path, 'rb') as f:
            src_data = pickle.load(f)

        # print('\n ==> SOURCE')
        # print('dshape:', src_data.shape)
        # print('col num: ', len(src_data.columns))
        # print('col', src_data.columns)

        filtered_data = src_data[columns] # parsing from colnb

        # print('\n ==> FILTER')
        # print('dshape:', filtered_data.shape)
        # print(filtered_data)

        # print('\n[-] \tparsing from {} ...\n'.format(self.src_path))

        return filtered_data
        

if __name__ == "__main__":
    methods = ['centroid', 'eoa', 'partial_pathlen', 'cumulate_pathlen', 'speed', 'velocity', 'IoU', 'gIoU']
    methods = ['centroid', 'partial_pathlen', 'IoU']
    methods = ['IoU', 'gIoU']
    extract_objs=['Grasper']
    extract_pairs=[('Grasper', 'Grasper')]

    src_path = '/dataset3/multimodal/PETRAW/Training/Seg_kine11/002_seg_ki.pkl'
    rk_filter = recon_kinematic_filter(task='PETRAW')
    rk_filter.set_src_path(src_path)
    filterd_data = rk_filter.filtering(methods, extract_objs, extract_pairs)
        
