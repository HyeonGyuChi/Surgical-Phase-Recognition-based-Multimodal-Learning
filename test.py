import os
import cv2
import json
import shutil
from operator import itemgetter


# base_path = '/dataset3/multimodal/val_imgs'
# file_list = os.listdir(base_path)

# base_path2 = '/dataset3/multimodal/val.json'

# with open(base_path2, 'r') as f:
#     data = json.load(f)


# id_dict = {}

# for d in data:
#     uid = d['product']
#     id_dict[uid] = 0


# for fname in file_list:
#     id_dict[fname] += 1


# for k, v in id_dict.items():
#     if v == 0:
#         print(k, v)


base_path = '/dataset3/multimodal/aliproducts2'

with open(base_path + '/train.json', 'r') as f:
    data = json.load(f)

id_dict = {}

for d in data:
    uid = d['product']
    id_dict[uid] = 0

for i in range(9):
    fpath = base_path + '/train_text_img_pairs_{}_compressed'.format(i)
    file_list = os.listdir(fpath)

    print(len(file_list))

    for fname in file_list:
        id_dict[fname] += 1

cnt = 0

for k, v in id_dict.items():
    if v == 0:
        print(k, v)
        cnt += 1
        if cnt > 10:
            break


