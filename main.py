import os
from datetime import datetime
from core.config.set_opts import load_opts
from core.api.trainer import Trainer
from core.api.inference import Predictor


def main():
    args = load_opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    now = datetime.now()
    current_time = now.strftime('%Y%m%d-%H%M%S')

    args.save_path = os.path.join(args.save_path, 'dset:{}-dtype:{}-model:{}-mdepth:{}-ipsize:{}-lfn:{}-lrs:{}-batch:{}-epoch:{}-{}'.format(args.dataset, \
                                                                                                    args.data_type, \
                                                                                                    args.model, \
                                                                                                    args.model_depth, \
                                                                                                    args.input_size, \
                                                                                                    args.loss_fn, \
                                                                                                    args.lr_scheduler, \
                                                                                                    args.batch_size, \
                                                                                                    args.max_epoch, \
                                                                                                    current_time))
    

    trainer = Trainer(args)
    trainer.fit()

    predictor = Predictor(args)
    predictor.inference()
    
if __name__ == '__main__':
    main()