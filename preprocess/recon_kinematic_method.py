import numpy as np

EXCEPTION_NUM = -100

def is_exception(x_min, x_max, y_min, y_max):
    if EXCEPTION_NUM in [x_min, x_max, y_min, y_max]:
        return True
    else:
        return False

def get_centroid(bbox_np): # (x min, x max, y min, y max)    
    centroid = np.array([EXCEPTION_NUM, EXCEPTION_NUM])

    x_min, x_max, y_min, y_max = bbox_np[0], bbox_np[1], bbox_np[2], bbox_np[3]
    
    if not is_exception(x_min, x_max, y_min, y_max):
        centroid = np.array([(x_max - x_min) / 2, (y_max - y_min) / 2]) 
        
    return centroid # numpy, (cen_x,cen_y)

def get_eoa(bbox_np): # (x min, x max, y min, y max)    
    eoa = np.array([EXCEPTION_NUM])

    x_min, x_max, y_min, y_max = bbox_np[0], bbox_np[1], bbox_np[2], bbox_np[3] 
    
    if not is_exception(x_min, x_max, y_min, y_max):
        eoa = (x_max - x_min) * (y_max - y_min) / 1.0

    return eoa # numpy, (eoa)

def get_speed(bbox_np): # (x min, x max, y min, y max)    
    # TODO - based from traj
    pass


def get_IoU(box1, box2):
    # TODO
    pass 

def get_gIoU(bbox1, bbox2):
    # TODO
    pass
