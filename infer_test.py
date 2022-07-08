import os
import pandas as pd
from core.config.set_opts import load_opts
from core.api.trainer import Trainer
from core.api.inference import Predictor


def main():
    args = load_opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    predictor = Predictor(args)
    metric = predictor.inference()

    print(metric)
    df = pd.DataFrame(metric)
    print(df)

    
if __name__ == '__main__':
    main()