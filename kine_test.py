import pickle
import numpy as np
import cv2
from glob import glob
import natsort


base_path = '/dataset3/multimodal/PETRAW/Training'
kine_path1 = base_path + '/Seg_kine'
kine_path2 = base_path + '/Seg_kine2'

kine_list1 = natsort.natsorted(glob(kine_path1 + '/*pkl'))
kine_list2 = natsort.natsorted(glob(kine_path2 + '/*pkl'))

idx = 0

with open(kine_list1[idx], 'rb') as f:
    data = pickle.load(f)

with open(kine_list2[idx], 'rb') as f:
    data2 = pickle.load(f)

diff = data - data2
diff_sum = np.sum(diff, 1)

ids = np.where(diff_sum == max(diff_sum))

print(diff_sum, max(diff_sum), ids)
print(data[ids[0], ])
print(data2[ids[0], ])


kine_path1 = base_path + '/Segmentation2/001/frame00001091.npz'

img = np.load(kine_path1)['arr_0']
print(img.shape)

cv2.imwrite('/code/test2.png', img)