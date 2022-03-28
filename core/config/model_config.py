def add_lstm_args(parser):
    parser.add_argument('--n_layer',
            default=1,
            type=int,)
    
    parser.add_argument('--linear_dim',
            default=256,
            type=int,)
    
    parser.add_argument('--hidden_size',
            default=256,
            type=int,)
    
    parser.add_argument('--use_bidirectional',
            action='store_true')
    
    return parser


def add_resnet_args(parser):
    parser.add_argument('--model_depth',
            default=18,
            type=int,)
    
    parser.add_argument('--n_input_channels',
            default=3,
            type=int,)
    
    parser.add_argument('--conv1_t_size',
            default=7,
            type=int,)
    
    parser.add_argument('--conv1_t_stride',
            default=1,
            type=int,)
    
    parser.add_argument('--shortcut_type',
            default='B',
            type=str,)
    
    parser.add_argument('--widen_factor',
            default=1.0,
            type=float,)
    
    parser.add_argument('--no_max_pool',
            action='store_true')
    
    
    return parser


def add_multi_args(parser):
    parser = add_lstm_args(parser)
    parser = add_resnet_args(parser)
    
    parser.add_argument('--model_params',
            default={
                    'video': {
                        'model': 'resnet3d',
                        'input_size': 224,
                        'feature_size': 512,
                        # 'restore_path': None,
                        'restore_path': 'logs/resnet3d-cb-loss/epoch:49-val_loss:7.8618.pth',
                    },
                    'kinematic': {
                        'model': 'lstm',
                        # 'input_size': 28,
                        # 'restore_path': 'logs/lstm-cb-loss/epoch:44-val_loss:8.2361.pth',
                        'input_size': 4,
                        # 'restore_path': 'logs/lstm-cb-loss-ski/epoch:49-val_loss:9.3289.pth',
                        'restore_path': 'logs/lstm-cb-loss-ski_swin/epoch:37-val_loss:9.2838.pth',
                        'feature_size': 256,
                    },
                    # other modality
                #     'mask': {
                #         'model': 'resent3d',
                #         'input_size': 224,
                #         'restore_path': None,
                #     },
                    },
            type=dict,)

    parser.add_argument('--fusion_type',
            default='conv',
    )

    parser.add_argument('--fusion_params',
            default={
                    'conv': {
                        'f_dim': 512,
                        'z_dim': 512,
                        'fusion_type': 'c',
                        'conv_sz': 3,
                        'use_pairwise': True,
                        'multi_scale': False,
                    }
            },
            type=dict,)

    return parser

def add_opts(parser):
    tmp_args = parser.parse_args()
    model_name = tmp_args.model
    
    if 'lstm' in model_name:
        parser = add_lstm_args(parser)
    elif 'resnet' in model_name:
        parser = add_resnet_args(parser)
    elif 'multi' in model_name:
        parser = add_multi_args(parser)
        
    return parser