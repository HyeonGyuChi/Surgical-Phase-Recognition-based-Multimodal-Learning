import os
from core.config.set_opts import load_opts
from core.api.trainer import Trainer


def main():
    args = load_opts()
    
    trainer = Trainer(args)
    
    trainer.fit()
    
if __name__ == '__main__':
    main()