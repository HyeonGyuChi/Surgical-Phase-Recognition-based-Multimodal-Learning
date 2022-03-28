import torch.optim as optim
import torch_optimizer as torch_optim # pytorch로 코딩된 optimizer들 모음


def configure_optimizer(config, model):
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    print('[+] Optimizer and Scheduler are set ', optimizer, scheduler)
    
    return optimizer, scheduler
    
def get_optimizer(config, model):
    # optimizer ref : https://pypi.org/project/torch-optimizer/
    
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(
                                model.parameters(),
                                lr=config.init_lr,
                                weight_decay=config.weight_decay,
                            )
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(
                                model.parameters(),
                                lr=config.init_lr,
                                weight_decay=config.weight_decay,
                            )
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(
                                model.parameters(),
                                lr=config.init_lr,
                                weight_decay=config.weight_decay,
                            )
    elif config.optimizer == 'adamp':
        optimizer = torch_optim.AdamP(
                                model.parameters(),
                                lr=config.init_lr,
                                weight_decay=config.weight_decay,
                                delta=0.1,
                                wd_ratio=0.1,
                            )
    elif config.optimizer == 'sgdw':
        optimizer = torch_optim.SGDW(
                                model.parameters(),
                                lr=config.init_lr,
                                weight_decay=config.weight_decay,
                                momentum=0,
                                dampening=0,
                                nesterov=False,
                            )
    elif config.optimizer == 'lamb':
        optimizer = torch_optim.Lamb(
                                model.parameters(),
                                lr=config.init_lr,
                                weight_decay=config.weight_decay,
                            )
        
        
    return optimizer

def get_scheduler(config, optimizer):
    schdlr_name = config.lr_scheduler
    
    if schdlr_name == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.lr_scheduler_step, 
            gamma=config.lr_scheduler_factor,
            verbose=True,
            )
    elif schdlr_name == 'mul_lr':
        scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimizer,
            lr_lambda=lambda epoch: config.lr_scheduler_factor,
            verbose=True,
        )
    elif schdlr_name == 'mul_step_lr':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.lr_milestones,
            gamma=config.lr_scheduler_factor,
            verbose=True,
        )
    elif schdlr_name == 'reduced_lr':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_step,
            threshold=1e-4,
            threshold_mode='rel',
            min_lr=1e-7,
            verbose=True,
        )
    elif schdlr_name == 'cosine_lr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epoch,
            eta_min=0,
            last_epoch=-1,
            verbose=True,
        )
        
    return scheduler