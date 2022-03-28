import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torch.utils.data import DataLoader
from core.config.set_opts import load_opts
import torchvision.models as models
from core.model import configure_optimizer, get_loss
from tqdm import tqdm
from core.dataset.petraw_seg_dataset import PETRAWDataset


PALETTE = [
            [0, 0, 255], # pegs -> 
            [255, 0, 0], # left_inst -> 1
            [255, 0, 255], # blocks -> 105 (3)
            [0, 255, 0], # right_inst -> 
            [255, 255, 255], # base -> 
            ]

def mapping(img):
    # img = img[:,:,0]
    new_img = np.zeros((*img.shape, 3))

    for i in range(len(PALETTE)):
        ids = np.where(img == i+1)
        new_img[ids[0], ids[1], ] = PALETTE[i]

    return new_img[:,:,::-1]


def dice_loss(pred, target, smooth = 1e-5):
    # binary cross entropy loss
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
    
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    # dice coefficient
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    
    # dice loss
    dice_loss = 1.0 - dice
    
    # total loss
    loss = bce + dice_loss
    
    return loss.sum(), dice.sum()

def dice(pred, target, smooth = 1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    # dice coefficient
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    
    return dice.sum()


def main():
    args = load_opts()
    args.loss_fn = 'ce'

    trainset = PETRAWDataset(args)
    valset = PETRAWDataset(args, state='valid')

    n_classes=7

    train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=6)
    val_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=6)
   
    # model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=n_classes, aux_loss=False)
    model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=21, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(256, n_classes, kernel_size=(1,1), stride=(1,1))
    model.aux_classifier[-1] = nn.Conv2d(256, n_classes, kernel_size=(1,1), stride=(1,1))

    optimizer, scheduler = configure_optimizer(args, model)
    loss_fn = get_loss(args)

    model = model.cuda()

    for epoch in range(1, 20):
        model.train()

        cnt = 0

        for data, lbs in tqdm(train_loader):
            data, lbs = data.cuda(), lbs.cuda()

            optimizer.zero_grad()

            output = model(data)
            out = output['out']
            
            l = lbs.cpu().data.numpy()
            # print(np.unique(l))

            loss = loss_fn(out.reshape(out.size(0), n_classes, -1), lbs.reshape(out.size(0), -1).long())
            aux = output['aux']
            loss += torch.mean(torch.mean(aux, dim=(1,2)))
 
            # print(loss)
            loss.backward()
            optimizer.step()

            if cnt > 50:
                break
            cnt += 1

            # break

        model.eval()

        with torch.no_grad():
            for data, lbs in val_loader:
                data, lbs = data.cuda(), lbs.cuda()

                output = model(data)
                out = output['out']

                o = torch.argmax(out, 1)
                o = o.cpu().data.numpy()

                # print(np.unique(o))

                for i in range(o.shape[0]):
                    seg = mapping(o[i, ])
                    cv2.imwrite('./Epoch:{}_{}_out.png'.format(epoch, i), seg)

                    if i > 3:
                        break

                break
    


if __name__ == '__main__':
    main()