import os
from glob import glob
import natsort
import numpy as np
import pickle
from tqdm import tqdm
import cv2

# segmentation2 : deeplabv3+
# segmentation3 : Swin
# segmentation4 : OCR

# Seg_kine : from deeplabv3+
# Seg_kine2 : from Swin
# Seg_kine3 : from OCR

base_path = '/dataset3/multimodal'
data_path = base_path + '/PETRAW/Training'
target_path = data_path + '/Segmentation2'
save_path = data_path + '/Seg_kine5'

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
    
    traj = np.zeros((len(frame_list), 10))
    
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
        else:
            right_x, right_y = -1, -1
            right_bbox = -1
        
        ids = np.where(new_img == 2)
        if len(ids[0]) > 0:
            left_x, left_y= np.mean(ids[1]), np.mean(ids[0])
            left_bbox = (np.max(ids[1]) - np.min(ids[1])) * (np.max(ids[0])-np.min(ids[0]))
        else:
            left_x, left_y = -1, -1
            left_bbox = -1
    
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

        if 'npz' == fpath[-3:]:
            traj[fi, :] = [left_x, left_y, right_x, right_y, 
                            l_x_speed, l_y_speed, r_x_speed, r_y_speed,
                            left_eoa, right_eoa,
                            ]
        else:
            traj[fi, :] = [right_x, right_y, left_x, left_y,
                            r_x_speed, r_y_speed, l_x_speed, l_y_speed,
                            right_eoa, left_eoa,
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