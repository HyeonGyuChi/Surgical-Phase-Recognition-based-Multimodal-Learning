import os
import natsort
from glob import glob
from tqdm import tqdm
import shutil



base_path = '/dataset3/multimodal/PETRAW/Training/Video'
tar_path = '/dataset3/multimodal/PETRAW/Training/test3'

dir_list = os.listdir(base_path)

cnt = 0
max_cnt = 1000
dir_idx = 1

n =0

for dir_name in dir_list:
    file_list = glob(os.path.join(base_path, dir_name) + '/*')

    n += len(file_list)
    print(n, file_list[0])

    for fpath in file_list:
        dpath = tar_path + '/img_{}'.format(dir_idx)
        os.makedirs(dpath, exist_ok=True)

        iname = fpath.split('/')[-1]
        dname = fpath.split('/')[-2]
        spath = dpath + '/{}_{}'.format(dname, iname)
        shutil.copy(fpath, spath)

        cnt += 1
        if cnt == max_cnt:
            cnt = 0
            dir_idx += 1


# src_path = base_path + '/img'

# img_list = natsort.natsorted(os.listdir(src_path))
# split = len(img_list) // 3

# tar_path = base_path + '/img2'
# for img_path in img_list[split:split*2]:
#     spath = os.path.join(src_path, img_path)
#     tpath = os.path.join(tar_path, img_path)
    
#     shutil.move(spath, tpath)

# tar_path = base_path + '/img3'
# for img_path in img_list[split*2:]:
#     spath = os.path.join(src_path, img_path)
#     tpath = os.path.join(tar_path, img_path)
    
#     shutil.move(spath, tpath)
