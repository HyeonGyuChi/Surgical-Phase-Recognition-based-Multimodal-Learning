import os
from datetime import datetime
from core.config.set_opts import load_opts
from core.api.trainer import Trainer
from core.api.inference import Predictor
from core.dataset import get_dataset

def main():
    args = load_opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    # trainer = Trainer(args)
    args.batch_size = 2

    train_loader, val_loader, ski_feature_num = get_dataset(args) # @HG modify

    d_loader = train_loader
    print(len(d_loader))

    data, labels = next(iter(d_loader))
    print(data['video'].shape, labels.shape)

    # trainer.fit()

    
if __name__ == '__main__':
    main()