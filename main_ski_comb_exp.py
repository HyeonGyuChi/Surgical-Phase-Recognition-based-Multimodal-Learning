import os
from core.config.set_opts import load_opts
from core.api.trainer import Trainer
from core.api.inference import Predictor

import itertools
import pickle


# pickle => csv
def log_to_table():
    metric_log_save_path = os.path.join('./new_5fps_logs_4','metric_log.pkl')
    with open(metric_log_save_path,'rb') as fw:
        metric_log = pickle.load(fw)

    best_acc = 0
    col = ['k', 'method', 'Phase', 'Step', 'Action_L', 'Action_R']
    table = []

    for k, v in metric_log.items():
        for k2, v2 in v.items():
            cnt = 0
            row = []
            row.append(k)
            row.append(k2)
            print('restore_path: ', v2['restore_path'])
            print('k: {}, method: {}'.format(k, k2))
            for v3 in v2['metric']:
                for k3, v4 in v3.items():
                    if k3 == 'Balance-Acc':
                        cnt+=1
                        row.append(v4)
                        
                        # print(np.array(row))
                        print(v4)   

            table.append(row)

            print(table)
            print('\n')

    table = np.array(table)
    table = pd.DataFrame(table, columns=col)
    table.to_csv(os.path.join(os.path.dirname(metric_log_save_path), 'results.csv'))
    print(table)


def ski_main(args): # experiment for ski
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    trainer = Trainer(args)
    args = trainer.fit() # set args.restore_path in trainer and args.input_size
    
    # load best epoch
    predictor = Predictor(args)
    metric = predictor.inference()

    return metric, args.restore_path # best model path

if __name__ == '__main__':

    total_methods=['centroid', 'eoa', 'partial_pathlen', 'cumulate_pathlen', 'speed', 'velocity', 'IoU', 'gIoU']

    args = load_opts()
    # for k in range(1, len(total_methods) + 1):

    combinations_of_methods = itertools.combinations(total_methods, args.k)
    combinations_of_methods = list(combinations_of_methods)
    
    print('\n\n k: {} => combination cnt: {}'.format(args.k, len(combinations_of_methods)))

    metric_log = {}
    metric_log[args.k] = {}
    
    # set ski_methods
    for ski_methods in combinations_of_methods:
        args = load_opts() # inits

        args.ski_methods = ski_methods

        print('\n\n====== exp setup ======')
        print('ski_methods: ', args.ski_methods)

        m, r_path = ski_main(args)

        if len(ski_methods) == 1:
            method_key = ski_methods[0]
        else:
            method_key = '-'.join(ski_methods)
            
        # save log
        metric_log[args.k][method_key] = {}
        metric_log[args.k][method_key]['restore_path'] = r_path
        metric_log[args.k][method_key]['metric'] = m

        # save log data
        metric_log_save_path = os.path.join(os.path.dirname(args.save_path), 'metric_log.pkl')
        with open(metric_log_save_path,'wb') as fw:
            pickle.dump(metric_log, fw)
    