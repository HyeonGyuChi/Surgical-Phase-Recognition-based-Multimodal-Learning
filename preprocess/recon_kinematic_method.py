import numpy as np
from numpy.lib.stride_tricks import sliding_window_view # over version 1.20.0
import math

EXCEPTION_NUM = -999

def is_exception(*argv):
    
    for arg in argv:
        if arg == EXCEPTION_NUM:
            return True

    return False

# for partial
def get_slide_window(nps, window_size = 8):
    d_len = nps.shape[0] # only 1 dim numpy

    pad_cnt = window_size - 1
    q, r = divmod(pad_cnt, 2) # quotient(몫), remainder(나머지)
    r = 1 if r > 0 else 0 # 0 or 1
    
    pre_pad_cnt = r + q
    pro_pad_cnt = q

    # zero padding
    pad_nps = np.concatenate((np.zeros(pre_pad_cnt), nps, np.zeros(pro_pad_cnt)), axis=0)

    # centroid view of windows
    window_nps = sliding_window_view(pad_nps, window_size)

    return window_nps # 2 dim (obj,window_size)

# for IoU series func
def outerbox(src_bbox_np, target_bbox_np):

    src_x_min, src_x_max, src_y_min, src_y_max = src_bbox_np[0], src_bbox_np[1], src_bbox_np[2], src_bbox_np[3]
    target_x_min, target_x_max, target_y_min, target_y_max = target_bbox_np[0], target_bbox_np[1], target_bbox_np[2], target_bbox_np[3]

    x_min = min(src_x_min, target_x_min)
    x_max = max(src_x_max, target_x_max)
    y_min = min(src_y_min, target_y_min)
    y_max = max(src_y_max, target_y_max)

    return x_min, x_max, y_min, y_max

# for IoU series func
def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


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

def get_partial_path_length(bbox_nps, window_size):
    # TODO - exception 
    f_len, _ = bbox_nps.shape
    path_len = np.zeros((f_len, 2)) # x_pathlen, y_pathlen

    disp = get_partial_displacement(bbox_nps, window_size=1)
    
    for f_idx in range(f_len):
        x_disp, y_disp = disp[f_idx, :]
        if not is_exception(x_disp, y_disp):
            path_len[f_idx, :] = np.abs(disp[f_idx, :]) # disp => path

        else:
            path_len[f_idx, :] = np.array([EXCEPTION_NUM] * 2, dtype=np.float64)
    
    go_partial = window_size > 1

    if go_partial:
        # partial path len
        x_pathlen, y_pathlen = path_len[:,0], path_len[:,1]
        x_pathlen_window, y_pathlen_window = get_slide_window(x_pathlen, window_size), get_slide_window(y_pathlen, window_size)
        
        for f_idx in range(f_len):
            if not np.any(x_pathlen_window[f_idx, :] == EXCEPTION_NUM):    
                path_len[f_idx, 0] = np.sum(x_pathlen_window[f_idx, :])
                path_len[f_idx, 1] = np.sum(y_pathlen_window[f_idx, :])
            else:
                path_len[f_idx, :] = np.array([EXCEPTION_NUM] * 2, dtype=np.float64)

    return path_len # numpy, (x_pathlen, y_pathlen) * f_len

def get_cumulate_path_length(bbox_nps):
    # TODO - exception 
    f_len, _ = bbox_nps.shape
    cum_path_len = np.zeros((f_len, 2)) # x_pathlen, y_pathlen
    path_len = get_partial_path_length(bbox_nps, window_size=1) # abs path

    print(path_len.dtype)

    cum_x_path, cum_y_path = 0, 0
    for f_idx in range(f_len):
        x_path, y_path = path_len[f_idx, :]
        if not is_exception(x_path, y_path):
            cum_x_path += path_len[f_idx, 0]
            cum_y_path += path_len[f_idx, 1]

        cum_path_len[f_idx, 0], cum_path_len[f_idx, 1] = cum_x_path, cum_y_path

    return cum_path_len # numpy, (cum_x_pathlen, cum_y_pathlen) * f_len

'''
def get_displacement(bbox_nps): # (x min, x max, y min, y max)
    f_len, _ = bbox_nps.shape
    disp = np.zeros((f_len, 2)) # x_pathlen, y_pathlen

    for f_idx in range(f_len):
        if f_idx == 0:
            x_pathlen, y_pathlen = 0, 0

        else:
            # mv up, mv right = postivie
            pre_cen_x, pre_cen_y = get_centroid(bbox_nps[f_idx-1, :])
            cen_x, cen_y = get_centroid(bbox_nps[f_idx, :])

            if not is_exception(pre_cen_x, pre_cen_y, cen_x, cen_y):
                x_pathlen, y_pathlen = cen_x - pre_cen_x, cen_y - pre_cen_y # for center
            
            else:
                x_pathlen, y_pathlen = EXCEPTION_NUM, EXCEPTION_NUM

            # traj = bbox_nps[f_idx, :] - bbox_nps[f_idx-1, :] # calc mv pixel
            # x_pathlen, y_pathlen = traj[1], traj[3] # x_max, y_max

        disp[f_idx, 0], disp[f_idx, 1] = x_pathlen, y_pathlen
    
    return disp # numpy, (x_pathlen, y_pathlen) * f_len
'''


def get_partial_displacement(bbox_nps, window_size): # (x min, x max, y min, y max)
    f_len, _ = bbox_nps.shape
    disp = np.zeros((f_len, 2)) # x_pathlen, y_pathlen

    for f_idx in range(f_len):
        if f_idx == 0:
            x_pathlen, y_pathlen = 0, 0

        else:
            # mv up, mv right = postivie
            pre_cen_x, pre_cen_y = get_centroid(bbox_nps[f_idx-1, :])
            cen_x, cen_y = get_centroid(bbox_nps[f_idx, :])

            if not is_exception(pre_cen_x, pre_cen_y, cen_x, cen_y):
                x_pathlen, y_pathlen = cen_x - pre_cen_x, cen_y - pre_cen_y # for center
            
            else:
                x_pathlen, y_pathlen = EXCEPTION_NUM, EXCEPTION_NUM

            # traj = bbox_nps[f_idx, :] - bbox_nps[f_idx-1, :] # calc mv pixel
            # x_pathlen, y_pathlen = traj[1], traj[3] # x_max, y_max
  
        disp[f_idx, 0], disp[f_idx, 1] = x_pathlen, y_pathlen

    go_partial = window_size > 1
    
    if go_partial:
        # partial path len
        x_disp, y_disp = disp[:,0], disp[:,1]
        x_disp_window, y_disp_window = get_slide_window(x_disp, window_size), get_slide_window(y_disp, window_size)
        
        for f_idx in range(f_len):
            if not np.any(x_disp_window[f_idx, :] == EXCEPTION_NUM):    
                disp[f_idx, 0] = np.sum(x_disp_window[f_idx, :])
                disp[f_idx, 1] = np.sum(y_disp_window[f_idx, :])
            else:
                disp[f_idx, :] = np.array([EXCEPTION_NUM] * 2, dtype=np.float64)
    
    return disp # numpy, (x_pathlen, y_pathlen) * f_len

def get_velocity(bbox_nps, interval_sec): # (x min, x max, y min, y max)
    # mv pixel/s = path length (pixel) / interval_sec 
    # TODO - exception 
    f_len, _ = bbox_nps.shape
    velocity = np.zeros((f_len, 2)) # x_pathlen, y_pathlen

    disp = get_partial_displacement(bbox_nps, window_size=1)
    
    for f_idx in range(f_len):
        if f_idx == 0:
            x_v, y_v = 0,0
        # interval_sec = 1/5 petraw
        # interval_sec = 1 gastric
        else:
            x_disp, y_disp = disp[f_idx, :]
            if not is_exception(x_disp, y_disp):
                x_v, y_v = x_disp / interval_sec, y_disp / interval_sec

            else:
                x_v, y_v = EXCEPTION_NUM, EXCEPTION_NUM
    
        velocity[f_idx, :] = x_v, y_v

    return velocity # numpy, (velocity) * f_len

def get_speed(bbox_nps, interval_sec): # (x min, x max, y min, y max)
    # mv pixel/s = path length (pixel) / interval_sec 
    # TODO - exception 
    f_len, _ = bbox_nps.shape
    speed = np.zeros((f_len, 1)) # x_pathlen, y_pathlen

    path_len = get_partial_path_length(bbox_nps, window_size=1)
    
    for f_idx in range(f_len):
        if f_idx == 0:
            s = 0

        else:
            x_pathlen, y_pathlen = path_len[f_idx, :]
            if not is_exception(x_pathlen, y_pathlen):
                s = np.sqrt((np.power(x_pathlen,2) + np.power(y_pathlen,2))) / interval_sec
            
            else:
                s = EXCEPTION_NUM
    
        speed[f_idx, :] = s

    return speed # numpy, (speed) * f_len

def get_IoU(src_bbox_np, target_bbox_np, return_U=False): # (x min, x max, y min, y max) / (x min, x max, y min, y max)

    IoU = EXCEPTION_NUM
    U = EXCEPTION_NUM

    src_x_min, src_x_max, src_y_min, src_y_max = src_bbox_np[0], src_bbox_np[1], src_bbox_np[2], src_bbox_np[3]
    target_x_min, target_x_max, target_y_min, target_y_max = target_bbox_np[0], target_bbox_np[1], target_bbox_np[2], target_bbox_np[3]

    if not is_exception(src_x_min, src_x_max, src_y_min, src_y_max) and \
            not is_exception(target_x_min, target_x_max, target_y_min, target_y_max):        
        
        # box = (x1, y1, x2, y2)
        src_area = (src_x_max - src_x_min + 1) * (src_y_max - src_y_min + 1)
        target_area = (target_x_max - target_x_min + 1) * (target_y_max - target_y_min + 1)

        # obtain x_min, max, y_min, max of the intersection
        x_min = max(src_x_min, target_x_min)
        x_max = min(src_x_max, target_x_max)
        y_min = max(src_y_min, target_y_min)
        y_max = min(src_y_max, target_y_max)

        # compute the width and height of the intersection
        w = max(0, x_max - x_min + 1)
        h = max(0, y_max - y_min + 1)

        inter = w * h
        U = src_area + target_area - inter + 1e-6 # error term 
        IoU = inter / U

    if return_U :
        iou = np.array([IoU, U])
    else:
        iou = np.array([IoU])

    return iou

def get_gIoU(src_bbox_np, target_bbox_np): # (x min, x max, y min, y max) / (x min, x max, y min, y max)
    # https://melona94.tistory.com/2

    gIoU = np.array([EXCEPTION_NUM])

    src_x_min, src_x_max, src_y_min, src_y_max = src_bbox_np[0], src_bbox_np[1], src_bbox_np[2], src_bbox_np[3]
    target_x_min, target_x_max, target_y_min, target_y_max = target_bbox_np[0], target_bbox_np[1], target_bbox_np[2], target_bbox_np[3]

    if not is_exception(src_x_min, src_x_max, src_y_min, src_y_max) and \
            not is_exception(target_x_min, target_x_max, target_y_min, target_y_max):   

        # outerbox : x min, max, y min, max
        outer_x_min, outer_x_max, outer_y_min, outer_y_max = outerbox(src_bbox_np, target_bbox_np)

        C = (outer_x_max - outer_x_min + 1) * (outer_y_max - outer_y_min + 1)

        IoU, U = get_IoU(src_bbox_np, target_bbox_np, return_U=True)

        gIoU = np.array([IoU - (C-U) / C])

    return gIoU

def get_dIoU(src_bbox_np, target_bbox_np): # (x min, x max, y min, y max) / (x min, x max, y min, y max)
    # https://cxybb.com/article/m0_53114462/117398110
    dIoU = np.array([EXCEPTION_NUM])

    src_x_min, src_x_max, src_y_min, src_y_max = src_bbox_np[0], src_bbox_np[1], src_bbox_np[2], src_bbox_np[3]
    target_x_min, target_x_max, target_y_min, target_y_max = target_bbox_np[0], target_bbox_np[1], target_bbox_np[2], target_bbox_np[3]

    if not is_exception(src_x_min, src_x_max, src_y_min, src_y_max) and \
            not is_exception(target_x_min, target_x_max, target_y_min, target_y_max):   
        
        # centroid
        src_cen_x, src_cen_y = get_centroid(src_bbox_np)
        target_cen_x, target_cen_y = get_centroid(target_bbox_np)

        # outerbox : x min, max, y min, max
        outer_x_min, outer_x_max, outer_y_min, outer_y_max = outerbox(src_bbox_np, target_bbox_np)

        # euclidan
        d = euclidean_distance((src_cen_x, src_cen_y), (target_cen_x, target_cen_y))
        c = euclidean_distance((outer_x_min, outer_y_min), (outer_x_max, outer_y_max)) + 1e-6 # error term

        # IoU
        IoU = get_IoU(src_bbox_np, target_bbox_np)

        dIoU = IoU - (d ** 2) / (c ** 2)
    
    return dIoU

def get_cIoU(src_bbox_np, target_bbox_np): # (x min, x max, y min, y max) / (x min, x max, y min, y max)
    # https://cxybb.com/article/m0_53114462/117398110
    cIoU = np.array([EXCEPTION_NUM])

    src_x_min, src_x_max, src_y_min, src_y_max = src_bbox_np[0], src_bbox_np[1], src_bbox_np[2], src_bbox_np[3]
    target_x_min, target_x_max, target_y_min, target_y_max = target_bbox_np[0], target_bbox_np[1], target_bbox_np[2], target_bbox_np[3]

    if not is_exception(src_x_min, src_x_max, src_y_min, src_y_max) and \
            not is_exception(target_x_min, target_x_max, target_y_min, target_y_max):   

        src_w, src_h = src_x_max - src_x_min + 1, src_y_max - src_y_min + 1
        target_w, target_h = target_x_max - target_x_min + 1, target_y_max - target_y_min + 1

        # aspect ratio
        v = 4 / (math.pi ** 2) * (math.atan(src_w / src_h) - math.atan(target_w/ target_h)) ** 2
        
        # IoU, dIoU
        IoU = get_IoU(src_bbox_np, target_bbox_np)
        dIoU = get_dIoU(src_bbox_np, target_bbox_np)

        alpha = v / (1 - IoU + v)

        cIoU = dIoU - alpha * v
    
    return cIoU

    
