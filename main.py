import os
from core.config.set_opts import load_opts
from core.api.trainer import Trainer
from core.api.inference import Predictor


def main():
    args = load_opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    trainer = Trainer(args)
    trainer.fit()

    predictor = Predictor(args)
    predictor.inference()
    
if __name__ == '__main__':
    main()