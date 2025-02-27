import os

from .dataset import FundusDataset
from .fundus_transforms import data_transforms
from utils.func import mean_and_std, print_dataset_info


def generate_dataset(cfg):
    dset = cfg.base.dataset
    if cfg.data.mean == 'auto' or cfg.data.std == 'auto':            
        mean, std = auto_statistics(cfg)        
        cfg.data.mean = mean  
        cfg.data.std = std     

    if dset in ['Fundus', 'AREDS']:
        train_transform, test_transform = data_transforms(cfg) 
        datasets = generate_fundus_dataset(cfg, train_transform, test_transform)  
    else:
        raise ArgumentError(f'Dataset not implemented: {cfg.base.dataset}')

    print_dataset_info(datasets)
    return datasets

def auto_statistics(cfg):
    input_size = cfg.data.input_size,
    batch_size = cfg.train.batch_size,
    num_workers = cfg.train.num_workers
    print('Calculating mean and std of training set for data normalization.')
    transform = simple_transform(input_size)

    #os.path.join(data_path, 'train') # initial 
    #train_dataset = datasets.ImageFolder(train_path, transform=transform)
    train_dataset = FundusDataset(cfg, transform=transform)

    return mean_and_std(train_dataset, batch_size, num_workers)

def generate_fundus_dataset(cfg, train_transform, test_transform):          
    dset_train = FundusDataset(cfg, transform=train_transform)
    dset_val = FundusDataset(cfg, train=False, transform=test_transform)
    dset_test = FundusDataset(cfg, train=False, test=True, transform=test_transform)
    return dset_train, dset_test, dset_val