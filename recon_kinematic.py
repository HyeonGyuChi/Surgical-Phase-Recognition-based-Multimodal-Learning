import os
from glob import glob
import natsort
import numpy as np
import pickle
from tqdm import tqdm
import cv2

base_path = '/dataset3/multimodal'
data_path = base_path + '/PETRAW/Training'
target_path = data_path + '/Segmentation3'
save_path = data_path + '/Seg_kine3'

os.makedirs(save_path, exist_ok=True)

file_list = natsort.natsorted(os.listdir(target_path))

N = 10
split = len(file_list) // N
st = 9

st_idx = st*split
ed_idx = (st+1)*split
if ed_idx > len(file_list):
    ed_idx = len(file_list)

file_list = file_list[st_idx:ed_idx]


for key_val in file_list:
    print(f'{key_val} start!')
    dpath = target_path + '/{}'.format(key_val)
    # frame_list = glob(dpath + '/*.npz')
    frame_list = glob(dpath + '/*.jpg')
    frame_list = natsort.natsorted(frame_list)
    
    traj = np.zeros((len(frame_list), 4))
    
    for fi, fpath in enumerate(tqdm(frame_list)):
        # img = np.load(fpath)['arr_0']
        img = cv2.imread(fpath)[:,:,::-1]
        h,w,c = img.shape
        
        new_img = np.zeros((h,w))
        
        ids = np.where(img[:,:,0]>0)
        if len(ids[0]) > 0:
            new_img[ids[0], ids[1]] = 1
        
        ids = np.where(img[:,:,1]>0)
        if len(ids[0]) > 0:
            new_img[ids[0], ids[1]] += 2
        
        ids = np.where(img[:,:,2]>0)
        if len(ids[0]) > 0:
            new_img[ids[0], ids[1]] += 4
        
        ids = np.where(new_img == 4)
        if len(ids[0]) > 0:
            right_x, right_y = np.mean(ids[1]), np.mean(ids[0])
        else:
            right_x, right_y = -1, -1
        
        ids = np.where(new_img == 2)
        if len(ids[0]) > 0:
            left_x, left_y= np.mean(ids[1]), np.mean(ids[0])
        else:
            left_x, left_y = -1, -1
    
        traj[fi, :] = [left_x, left_y, right_x, right_y]
    
    for ti in range(len(traj)):
        info = traj[ti, :]
        
        # if np.isnan(info).any():
        #     print(ti, np.isnan(info))
        #     img = np.load(frame_list[ti])['arr_0']
            
        #     ids = np.where(img[:,:,0]>0)
        #     ids2 = np.where(img[:,:,1]>0)
        #     ids3 = np.where(img[:,:,2]>0)
            
        #     print(len(ids[0]), len(ids2[0]), len(ids3[0]))
            
        #     cv2.imwrite(f"./test_{ti}.png", img)
            
        
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
        
    
    # print('a : ', len(np.array(traj == -1)[0]))
    # print('b : ', len(np.isnan(traj)))
    
    # break
        
    save_path2 = save_path + '/{}_seg_ki.pkl'.format(key_val)
    with open(save_path2, 'wb') as f:
        pickle.dump(traj, f)