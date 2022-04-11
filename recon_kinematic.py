import os
from glob import glob
import natsort
import numpy as np
import pickle
from tqdm import tqdm
import cv2


def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    U = (box1_area + box2_area - inter)

    return iou, U

def gIoU(bbox1, bbox2):
    # https://melona94.tistory.com/2
    # input : [x, y, w, h] (bbox) x 2
    bbox1 = [*bbox1[:2], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]]
    bbox2 = [*bbox2[:2], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]]

    # min x
    if bbox1[0] < bbox2[0]:
        l_x = bbox1[0]
    else:
        l_x = bbox2[0]

    # max x
    if bbox1[2] < bbox2[2]:
        r_x = bbox2[2]
    else:
        r_x = bbox1[2]

    # min y
    if bbox1[1] < bbox2[1]:
        l_y = bbox1[1]
    else:
        l_y = bbox2[1]

    # max y
    if bbox1[3] < bbox2[3]:
        r_y = bbox2[3]
    else:
        r_y = bbox1[3]

    C = (r_x-l_x+1) * (r_y-l_y+1)

    iou, U = IoU(bbox1, bbox2)

    giou = iou - (C-U)/C

    # if np.isinf(giou):
    #     print(bbox1)
    #     print(bbox2)
    #     print(l_x, r_x, l_y, r_y)
    #     print(C, iou, U, giou)
    #     print()

    return giou


# segmentation2 : deeplabv3+
# segmentation3 : Swin
# segmentation4 : OCR

# Seg_kine : from deeplabv3+
# Seg_kine2 : from Swin
# Seg_kine3 : from OCR

base_path = '/dataset3/multimodal'
data_path = base_path + '/PETRAW/Training'
target_path = data_path + '/Segmentation3'
save_path = data_path + '/Seg_kine7'

os.makedirs(save_path, exist_ok=True)

file_list = natsort.natsorted(os.listdir(target_path))

N = 10
split = len(file_list) // N
st = 0

st_idx = st*split
ed_idx = (st+1)*split
if ed_idx > len(file_list):
    ed_idx = len(file_list)

file_list = file_list[st_idx:ed_idx]

for key_val in file_list:
    print(f'{key_val} start!')
    dpath = target_path + '/{}'.format(key_val)
    frame_list = glob(dpath + '/*')
    frame_list = natsort.natsorted(frame_list)
    
    traj = np.zeros((len(frame_list), 11))
    
    for fi, fpath in enumerate(tqdm(frame_list)):
        if 'npz' == fpath[-3:]:
            img = np.load(fpath)['arr_0']
        else:
            img = cv2.imread(fpath)
            img = cv2.resize(img, dsize=(512, 512))

        h,w,c = img.shape
        new_img = np.zeros((h,w))
        
        ids = np.where(img[:,:,0]>0) # blue
        if len(ids[0]) > 0:
            new_img[ids[0], ids[1]] = 1
        
        ids = np.where(img[:,:,1]>0) # green
        if len(ids[0]) > 0:
            new_img[ids[0], ids[1]] += 2
        
        ids = np.where(img[:,:,2]>0) # red
        if len(ids[0]) > 0:
            new_img[ids[0], ids[1]] += 4
        
        ids = np.where(new_img == 4)
        if len(ids[0]) > 0:
            right_x, right_y = np.mean(ids[1]), np.mean(ids[0])
            right_bbox = (np.max(ids[1]) - np.min(ids[1])) * (np.max(ids[0])-np.min(ids[0]))

            r_bbox = [np.min(ids[1]), np.max(ids[1]), np.min(ids[0]), np.max(ids[0])]
        else:
            right_x, right_y = -1, -1
            right_bbox = -1
            r_bbox = -1
        
        ids = np.where(new_img == 2)
        if len(ids[0]) > 0:
            left_x, left_y= np.mean(ids[1]), np.mean(ids[0])
            left_bbox = (np.max(ids[1]) - np.min(ids[1])) * (np.max(ids[0])-np.min(ids[0]))

            l_bbox = [np.min(ids[1]), np.max(ids[1]), np.min(ids[0]), np.max(ids[0])]
        else:
            left_x, left_y = -1, -1
            left_bbox = -1
            l_bbox = -1
    
        if left_bbox != -1:
            left_eoa = left_bbox/(512*512)
        if right_bbox != -1:
            right_eoa = right_bbox/(512*512)

        if fi == 0:
            l_x_speed, l_y_speed = 0, 0
            r_x_speed, r_y_speed = 0, 0
        else:
            l_x_speed, l_y_speed = (r_x_speed-traj[fi-1, 4]), (r_y_speed-traj[fi-1, 5])
            r_x_speed, r_y_speed = (l_x_speed-traj[fi-1, 6]), (l_y_speed-traj[fi-1, 7])

        if l_bbox != -1 and r_bbox != -1:
            giou = gIoU(l_bbox, r_bbox)
        else:
            giou = -1

        if 'npz' == fpath[-3:]:
            traj[fi, :] = [left_x, left_y, right_x, right_y, 
                            l_x_speed, l_y_speed, r_x_speed, r_y_speed,
                            left_eoa, right_eoa, giou,
                            ]
        else:
            traj[fi, :] = [right_x, right_y, left_x, left_y,
                            r_x_speed, r_y_speed, l_x_speed, l_y_speed,
                            right_eoa, left_eoa, giou,
                            ]

    for ti in range(len(traj)):
        info = traj[ti, :]
        
        for di in range(len(info)):
            if np.isnan(info[di]):
                if ti == 0:
                    for idx in range(1, len(traj)):
                        if info[ti+idx, di] != -1:
                            info[di] = info[ti+idx, di]
                            break
                elif ti+1 == len(traj):
                    for idx in range(1, len(traj)):
                        if info[ti-idx, di] != -1:
                            info[di] = info[ti-idx, di]
                            break
                else:
                    l_idx = 1
                    r_idx = 1
                    l_val = -1
                    r_val = -1
                    
                    for l_idx in range(ti-1, -1, -1):
                        if info[l_idx, di] != -1:
                            l_val = info[l_idx, di]
                            break
                        
                    for r_idx in range(ti+1, len(traj)):
                        if info[r_idx, di] != -1:
                            r_val = info[r_idx, di]
                            break
                    
                    if r_val != -1 and l_val != -1:
                        info[di] = (l_val + r_val) / 2.
                    elif r_val == -1 and l_val != -1:
                        info[di] = l_val
                    elif r_val != -1 and l_val == -1:
                        info[di] = r_val
    
        traj[ti, :] = info
        
    save_path2 = save_path + '/{}_seg_ki.pkl'.format(key_val)
    with open(save_path2, 'wb') as f:
        pickle.dump(traj, f)