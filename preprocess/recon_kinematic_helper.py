from recon_kinematic_method import get_centroid, get_eoa, get_path_length, get_velocity, get_IoU, get_gIoU
from loader import PETRAWBBOXLoader

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


def get_recon_method(method):
    recon_method = {
        'centroid': get_centroid,
        'eoa': get_eoa,
        'pathlen': get_path_length,
        'velocity': get_velocity,
        'IoU': get_IoU,
        'gIoU': get_gIoU,
    }

    recon_method_col = {
        'centroid': ['x_centroid', 'y_centorid'],
        'eoa': ['eoa'],
        'pathlen': ['x_pathlen', 'y_pathlen'],
        'velocity': ['velocity'],
        'IoU': ['IoU', 'U'],
        'gIoU': ['gIoU'],
    }

    return recon_method[method], recon_method_col[method]
