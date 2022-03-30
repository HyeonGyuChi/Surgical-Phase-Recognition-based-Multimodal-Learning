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

with open(kine_list1[0], 'rb') as f:
    data = pickle.load(f)

with open(kine_list2[0], 'rb') as f:
    data2 = pickle.load(f)


print(kine_list1[0])
print(kine_list2[0])
print(data.shape, data2.shape)

print(data[:10, ])
print(data2[:10, ])