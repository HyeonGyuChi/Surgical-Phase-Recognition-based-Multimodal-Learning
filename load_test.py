import os
import numpy as np
import cv2
import torch
import pickle
import gzip
from tqdm import tqdm

PALETTE = [[0, 0, 0],
                [251, 244, 5], [37, 250, 5],[0, 21, 209],
                [172, 21, 2],[172, 21, 229],
                [6, 254, 249],[141, 216, 23],[96, 13, 13],
                [65, 214, 24],[124, 3, 252],[214, 55, 153],[48, 61, 173],
                [110, 31, 254],[249, 37, 14],[249, 137, 254],
                [34, 255, 113],[169, 52, 14],
                [124, 49, 176],[4, 88, 238],
                [115, 214, 178],[115, 63, 178],
                [115, 214, 235],[63, 63, 178],
                [130, 34, 26],[220, 158, 161],
                [201, 117, 56],[121, 16, 40],
                [15, 126, 0],[224, 224, 224],
                [154, 0, 0],[204, 102, 0]]


def mapping(seg):
    # palette = np.array(PALETTE)

    # color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    # for label, color in enumerate(palette):
    #     color_seg[seg == label, :] = color
    # # convert to BGR
    # color_seg = color_seg[..., ::-1]

    # return color_seg

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 32), dtype=np.uint8)
    for label in range(32):
        color_seg[seg == label, label] = 1

    return color_seg

    # new_img = np.zeros((512, 512, 3))

    # for lb in range(32):
    #     ids = np.where(img == lb)

    #     if len(ids[0]) > 0:
    #         new_img[ids[0], ids[1], 2] = PALETTE[lb][0]
    #         new_img[ids[0], ids[1], 1] = PALETTE[lb][1]
    #         new_img[ids[0], ids[1], 0] = PALETTE[lb][2]

    # return new_img


def main():
    base_path = '/dataset3/multimodal/gastric/Segmentation/R000001/ch1_video_01/frame0000000007.gz'
        
    with gzip.open(base_path, 'rb') as f:
        data = pickle.load(f)

    # base_path = '/code/R000001_ch1_video_01_0000000001.png'
    # data = cv2.imread(base_path)

    print(data.shape)

    for i in range(32):
        img = data[:,:,i]
        ids = np.where(img>0)
        print(len(ids[0]))
        print(np.max(np.max(img)))
        img[img > 0] = 255
        img[img < 255] = 0

        cv2.imwrite('/code/img_{}.png'.format(i), img)

    
if __name__ == '__main__':
    main()