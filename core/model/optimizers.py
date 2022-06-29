import torch.optim as optim
import torch_optimizer as torch_optim # pytorch로 코딩된 optimizer들 모음
from warmup_scheduler import GradualWarmupScheduler


def configure_optimizer(args, model):
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    print('[+] Optimizer and Scheduler are set ', optimizer, scheduler)
    
    return optimizer, scheduler
    
def get_optimizer(args, model):
    # optimizer ref : https://pypi.org/project/torch-optimizer/
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
                                model.parameters(),
                                lr=args.init_lr,
                                weight_decay=args.weight_decay,
                            )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
                                model.parameters(),
                                lr=args.init_lr,
                                weight_decay=args.weight_decay,
                            )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
                                model.parameters(),
                                lr=args.init_lr,
                                weight_decay=args.weight_decay,
                            )
    elif args.optimizer == 'adamp':
        optimizer = torch_optim.AdamP(
                                model.parameters(),
                                lr=args.init_lr,
                                weight_decay=args.weight_decay,
                                delta=0.1,
                                wd_ratio=0.1,
                            )
    elif args.optimizer == 'sgdw':
        optimizer = torch_optim.SGDW(
                                model.parameters(),
                                lr=args.init_lr,
                                weight_decay=args.weight_decay,
                                momentum=0,
                                dampening=0,
                                nesterov=False,
                            )
    elif args.optimizer == 'lamb':
        optimizer = torch_optim.Lamb(
                                model.parameters(),
                                lr=args.init_lr,
                                weight_decay=args.weight_decay,
                            )
        
        
    return optimizer

def get_scheduler(args, optimizer):
    schdlr_name = args.lr_scheduler
    
    if schdlr_name == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_scheduler_step, 
            gamma=args.lr_scheduler_factor,
            verbose=True,
            )
    elif schdlr_name == 'mul_lr':
        scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimizer,
            lr_lambda=lambda epoch: args.lr_scheduler_factor,
            verbose=True,
        )
    elif schdlr_name == 'mul_step_lr':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.lr_milestones,
            gamma=args.lr_scheduler_factor,
            verbose=True,
        )
    elif schdlr_name == 'reduced_lr':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_step,
            threshold=1e-4,
            threshold_mode='rel',
            min_lr=1e-7,
            verbose=True,
        )
    elif schdlr_name == 'cosine_lr':
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epoch,
            eta_min=0,
            last_epoch=-1,
            verbose=True,
        )

        scheduler = GradualWarmupScheduler(optimizer, 
                                            multiplier=1., 
                                            total_epoch=34, 
                                            after_scheduler=base_scheduler
                                        )

    return scheduler