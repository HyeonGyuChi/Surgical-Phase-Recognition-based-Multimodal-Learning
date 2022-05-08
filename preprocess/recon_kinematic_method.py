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

def get_eoa(bbox_np, img_size): # (x min, x max, y min, y max)    
    w, h = img_size 
    eoa = np.array([EXCEPTION_NUM])

    x_min, x_max, y_min, y_max = bbox_np[0], bbox_np[1], bbox_np[2], bbox_np[3] 
    
    if not is_exception(x_min, x_max, y_min, y_max):
        eoa = np.array([(x_max - x_min) * (y_max - y_min) / w*h])

    return eoa # numpy, (eoa)

def get_partial_path_length(bbox_nps):
    # TODO - exception 
    f_len, _ = bbox_nps.shape
    path_len = np.zeros((f_len, 2)) # x_pathlen, y_pathlen

    for f_idx in range(f_len):
        if f_idx == 0:
            x_pathlen, y_pathlen = 0, 0

        else:
            # mv up, mv right = postivie
            pre_cen_x, pre_cen_y = get_centroid(bbox_nps[f_idx-1, :])
            cen_x, cen_y = get_centroid(bbox_nps[f_idx, :])

            x_pathlen, y_pathlen = cen_x - pre_cen_x, cen_y - pre_cen_y # for center

            # traj = bbox_nps[f_idx, :] - bbox_nps[f_idx-1, :] # calc mv pixel
            # x_pathlen, y_pathlen = traj[1], traj[3] # x_max, y_max

        path_len[f_idx, 0], path_len[f_idx, 1] = x_pathlen, y_pathlen

    return path_len # numpy, (x_pathlen, y_pathlen) * f_len

def get_cumulate_path_length(bbox_nps):
    # TODO - exception 
    f_len, _ = bbox_nps.shape
    cum_path_len = np.zeros((f_len, 2)) # x_pathlen, y_pathlen
    path_len = np.abs(get_partial_path_length(bbox_nps)) # abs path

    for f_idx in range(f_len):
        cum_path_len[f_idx, :] += path_len[f_idx, :]

    return cum_path_len # numpy, (cum_x_pathlen, cum_y_pathlen) * f_len


def get_velocity(bbox_nps, interval_sec): # (x min, x max, y min, y max)
    # mv pixel/s = path length (pixel) / interval_sec 
    # TODO - exception 
    f_len, _ = bbox_nps.shape
    velocity = np.zeros((f_len, 1)) # x_pathlen, y_pathlen

    path_len = get_partial_path_length(bbox_nps)
    
    for f_idx in range(f_len):
        if f_idx == 0:
            v = 0
        
        else:
            x_pathlen, y_pathlen = path_len[f_idx,0], path_len[f_idx,1]
            v = (np.abs(x_pathlen) + np.abs(y_pathlen)) / interval_sec
    
        velocity[f_idx] = v

    return velocity # numpy, (velocity) * f_len

def get_IoU(src_bbox_np, target_bbox_np): # (x min, x max, y min, y max) / (x min, x max, y min, y max)

    IoU = EXCEPTION_NUM
    U = EXCEPTION_NUM
    np.array([IoU, U])

    src_x_min, src_x_max, src_y_min, src_y_max = src_bbox_np[0], src_bbox_np[1], src_bbox_np[2], src_bbox_np[3]
    target_x_min, target_x_max, target_y_min, target_y_max = target_bbox_np[0], target_bbox_np[1], target_bbox_np[2], target_bbox_np[3]

    if not is_exception(src_x_min, src_x_max, src_y_min, src_y_max) and \
            not is_exception(target_x_min, target_x_max, target_y_min, target_y_max):        
        
        # box = (x1, y1, x2, y2)
        src_area = (src_x_max - src_x_min + 1) * (src_y_max - src_y_min + 1)
        target_area = (target_x_max - target_x_min + 1) * (src_y_max - target_y_min + 1)

        # obtain x_min, max, y_min, max of the intersection
        x_min = max(src_x_min, target_x_min)
        x_max = min(src_x_max, target_x_max)
        y_min = max(src_y_min, target_y_min)
        y_max = min(src_y_max, target_y_max)

        # compute the width and height of the intersection
        w = max(0, x_max - x_min + 1)
        h = max(0, y_max - y_min + 1)

        inter = w * h
        IoU = inter / (src_area + target_area - inter)
        U = (src_area + target_area - inter)

    return np.array([IoU, U])

def get_gIoU(src_bbox_np, target_bbox_np): # (x min, x max, y min, y max) / (x min, x max, y min, y max)
    # https://melona94.tistory.com/2

    gIoU = np.array([EXCEPTION_NUM])

    src_x_min, src_x_max, src_y_min, src_y_max = src_bbox_np[0], src_bbox_np[1], src_bbox_np[2], src_bbox_np[3]
    target_x_min, target_x_max, target_y_min, target_y_max = target_bbox_np[0], target_bbox_np[1], target_bbox_np[2], target_bbox_np[3]

    if not is_exception(src_x_min, src_x_max, src_y_min, src_y_max) and \
            not is_exception(target_x_min, target_x_max, target_y_min, target_y_max):   

        # x min, max, y min, max
        x_min = src_x_min if src_x_min < target_x_min else target_x_min
        x_max = src_x_max if src_x_max < target_x_max else target_x_max
        y_min = src_y_min if src_y_min < target_y_min else target_y_min
        y_max = src_y_max if src_y_max < target_y_max else target_y_max

        C = (x_max - x_min + 1) * (y_max - y_min + 1)

        IoU, U = get_IoU(src_bbox_np, target_bbox_np)

        gIoU = np.array([IoU - (C-U) / C])

    return gIoU