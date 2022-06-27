import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from core.config.set_opts import load_opts
from core.model import get_model
from core.dataset import get_dataset

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
    palette = np.array(PALETTE)

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

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
    import pickle
    import gzip

    save_path = '/dataset3/multimodal/gastric/Segmentation/R000001/ch1_video_01/frame0000124789.gz'
    # data = np.load(save_path)
    with gzip.open(save_path, 'rb') as f:
        data = pickle.load(f)
    print(data.shape)
    # args = load_opts()
    # args.dataset = 'infer_gast'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    # model = get_model(args)
    # model = model.cuda()
    # model.eval()
    # train_loader, val_loader = get_dataset(args, return_ski_feature_num=False) # @HG modify
    
    # with torch.no_grad():
    #     for data in tqdm(train_loader):
    #         li, imgs, lbs = data
    #         img_metas = [
    #             {
    #                 'filename': li,
    #                 'img_shape': [512, 512, 3],
    #                 'scale_factor': [1],
    #                 'flip': False,
    #                 'ori_shape': [512, 512, 3],
    #                 'pad_shape': [512, 512, 3],
    #                 # 'img_'
    #             }
    #         ]
    #         imgs = imgs['video'].cuda()
    #         output = model.encode_decode(imgs, img_metas)
    #         # output = model.forward([imgs], img_metas, return_loss=False)
    #         output = torch.argmax(output, dim=1)

    #         for i in range(output.shape[0]):
    #             # out = mapping(output[i, ].cpu().data.numpy())
    #             out = output[i, ].cpu().data.numpy()

    #             img_path = li[i]
    #             tokens = img_path.split('/')

    #             save_path = '/dataset3/multimodal/gastric/Segmentation' + '/{}/{}'.format(*tokens[-3:-1])
    #             os.makedirs(save_path, exist_ok=True)
    #             save_path2 = save_path + '/{}'.format(tokens[-1].split('.')[0])
    #             # cv2.imwrite(save_path2, out)
    #             np.save(save_path2, out)

    #         break


    
if __name__ == '__main__':
    main()