from torch.utils.data import DataLoader

from core.dataset.jigsaws_dataset import JIGSAWSDataset
from core.dataset.misaw_dataset import MISAWDataset
from core.dataset.petraw_dataset import PETRAWDataset
from torchvision.datasets import MNIST, CIFAR10


dataset_dict = {
    'mnist': None,
    'petraw': PETRAWDataset,
    'misaw': MISAWDataset,
    'jigsaws': JIGSAWSDataset,
}


def get_dataset(args):
    if args.dataset == 'mnist':
        import torchvision.transforms as transforms
        
        tt = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        
        trainset = CIFAR10(args.data_base_path, train=True, download=True, transform=tt)
        valset = CIFAR10(args.data_base_path, train=False, download=True, transform=tt)
    else:
        trainset = dataset_dict[args.dataset](args, state='train')
        valset = dataset_dict[args.dataset](args, state='valid')
    
    train_loader = DataLoader(trainset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True)

    val_loader = DataLoader(valset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)
        
    return train_loader, val_loader

