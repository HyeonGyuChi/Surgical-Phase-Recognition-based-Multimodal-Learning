from recon_kinematic_method import get_centroid, get_eoa, get_partial_path_length, get_cumulate_path_length, get_velocity, get_IoU, get_gIoU
from loader import PETRAWBBOXLoader


def normalized_pixel(pixel_np, size):
    return pixel_np / size

def denormalized_pixel(pixel_np, size):  
    return pixel_np * size


def get_bbox_loader(task, target_path, dsize):
    dataloader = {
        'PETRAW': PETRAWBBOXLoader,
    }

    return dataloader[task](target_path, dsize)

def set_bbox_loader(bbox_loader, target_path, dsize):
    bbox_loader.set_root_dir(target_path)
    bbox_loader.set_dsize(dsize)

    return bbox_loader

def get_bbox_obj_info(bbox_loader):
    obj_key, obj_to_color = bbox_loader.obj_key, bbox_loader.obj_to_color
    
    return obj_key, obj_to_color


def get_recon_method(method, img_size):
    recon_method = {
        'centroid': get_centroid,
        'eoa': get_eoa,
        'partial_pathlen': get_partial_path_length,
        'cumulate_pathlen': get_cumulate_path_length,
        'velocity': get_velocity,
        'IoU': get_IoU,
        'gIoU': get_gIoU,
    }

    recon_method_col = {
        'centroid': ['x_centroid', 'y_centorid'], # ==> w, h
        'eoa': ['eoa'], # ==> w*h
        'partial_pathlen': ['x_p_pathlen', 'y_p_pathlen'], # ==> w, h
        'cumulate_pathlen': ['x_c_pathlen', 'y_c_pathlen'], # ==> w, h
        'velocity': ['velocity'],
        'IoU': ['IoU', 'U'],
        'gIoU': ['gIoU'],
    }

    w, h = img_size
    normalized_weight = {
        'centroid': [w, h], # ==> w, h
        'eoa': [1], # no normalized 
        'partial_pathlen': [w, h], # ==> w, h
        'cumulate_pathlen': [w, h], # ==> w, h
        'velocity': [1], # no normalized
        'IoU': [1],
        'gIoU': [1],
    }

    return recon_method[method], recon_method_col[method], normalized_weight[method]
