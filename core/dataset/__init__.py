from torch.utils.data import DataLoader

from core.dataset.jigsaws_dataset import JIGSAWSDataset
from core.dataset.misaw_dataset import MISAWDataset
# from core.dataset.petraw_dataset import PETRAWDataset
from core.dataset.gast_dataset import GastrectomyDataset
from core.dataset.gast_dataset_infer import InferGastrectomyDataset
from core.dataset.gast_dataset_mmver import GastrectomyDatasetMM
from torchvision.datasets import MNIST, CIFAR10


dataset_dict = {
    'mnist': None,
    # 'petraw': PETRAWDataset,
    'misaw': MISAWDataset,
    'jigsaws': JIGSAWSDataset,
    'gast': GastrectomyDataset,
    'gast_mm': GastrectomyDatasetMM,
    'infer_gast': InferGastrectomyDataset,
    'gast_mm': GastrectomyDatasetMM,
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
                            num_workers=args.num_workers * args.num_gpus,
                            pin_memory=True)

    val_loader = DataLoader(valset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers * args.num_gpus,
                            pin_memory=True)

    num_of_ski_feature = 0

    if hasattr(trainset,'num_of_ski_feature'):
        num_of_ski_feature = getattr(trainset, 'num_of_ski_feature')
    
    return train_loader, val_loader, num_of_ski_feature

