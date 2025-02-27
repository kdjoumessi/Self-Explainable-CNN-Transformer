import os
import yaml
import torch
import shutil
import socket
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from datetime import datetime
from tqdm import tqdm
from munch import munchify
from torch.utils.data import DataLoader

from utils.const import regression_loss

CIN_hostname = []

def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        '-config',
        type=str,
        default='./configs/default.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-paths',
        type=str,
        default='./configs/paths.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-overwrite',
        action='store_true',
        default=False,
        help='Overwrite file in the save path.'
    )
    parser.add_argument(
        '-print_config',
        action='store_true',
        default=False,
        help='Print details of configs.'
    )
    args = parser.parse_args()
    return args

def load_save_paths(cfg):
    timestamp_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")  
    save_path_model = 'Outputs/tmp/BagNet_SA' if cfg.base.test else f'Outputs/BagNet_SA/{cfg.base.dataset}'
    save_path = os.path.join(os.path.expanduser('~'), save_path_model, timestamp_str) 
    return save_path

def data_path(cfg, cfg_path):
    dset = cfg.base.dataset
    
    cfg.data.std = cfg_path[dset].std 
    cfg.data.mean = cfg_path[dset].mean
    
    cfg.dset['val_csv'] = cfg_path[dset].val_csv
    cfg.dset['test_csv'] = cfg_path[dset].test_csv
    cfg.dset['train_csv'] = cfg_path[dset].train_csv      

    cfg.data['fname'] = cfg_path[dset].fname 
    cfg.data['binary'] = cfg_path[dset].binary
    cfg.data['target'] = cfg_path[dset].target
    cfg.data['threshold'] = cfg_path[dset].threshold
    cfg.data['input_size'] = cfg_path[dset].input_size    
    cfg.data['num_classes'] = cfg_path[dset].num_classes 
    
    cfg.dset['root'] = cfg_path[dset].root
    cfg.dset['data_dir'] = cfg_path[dset].data_dir
    return cfg

def load_config(args):
    with open(args.config, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    with open(args.paths, 'r') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    return munchify(cfg), munchify(paths) 

def load_conf_file(config_file_path):
    '''
        Load the conf file containing all the parameters with the dataset paths

        input:
            - config_file_path (str): path to the configuration file

        output
            - a dictionary containing the conf parameters
    '''
    with open(config_file_path) as fhandle:
        cfg = yaml.safe_load(fhandle)
        
    cfg = munchify(cfg)
    return cfg


def copy_config(src, dst):
    shutil.copy(src, dst)


### -----------------------------
def copy_file(folds, dest_dir):
    # copy files: can be more customize
    for directory in folds:
        src_dir = os.path.join(os.getcwd(), directory)
        dst_dir = os.path.join(dest_dir, directory)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        files = os.listdir(src_dir)
        for file in files:
            fpath = os.path.join(src_dir, file)
            if file.startswith("__"):
                continue
            elif os.path.isdir(fpath):
                fs = os.listdir(fpath)
                for f in fs:
                    if not f.startswith(tuple(["__", "."])):
                        if os.path.isdir(os.path.join(fpath, f)):
                            continue ## norms and aother
                        shutil.copy2(os.path.join(fpath, f), dst_dir)
            else:
                shutil.copy2(fpath, dst_dir)
    
    #shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=dest_dir)    
    shutil.copy(src=os.path.join(os.getcwd(), 'train.py'), dst=dest_dir)    
    shutil.copy(src=os.path.join(os.getcwd(), 'main.py'), dst=dest_dir)


def save_config(config, path):
    with open(path, 'w') as file:
        yaml.safe_dump(config, file)


def mean_and_std(train_dataset, batch_size, num_workers):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    num_samples = 0.
    channel_mean = torch.Tensor([0., 0., 0.])
    channel_std = torch.Tensor([0., 0., 0.])
    for samples in tqdm(loader):
        X, _ = samples
        channel_mean += X.mean((2, 3)).sum(0)
        num_samples += X.size(0)
    channel_mean /= num_samples

    for samples in tqdm(loader):
        X, _ = samples
        batch_samples = X.size(0)
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples
    channel_std = torch.sqrt(channel_std / num_samples)

    mean, std = channel_mean.tolist(), channel_std.tolist()
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    return mean, std


def save_weights(model, save_path, mod=False):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()    
    torch.save(state_dict, save_path)
    
    if mod:
        torch.save(obj=model, f=save_path)


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)


def print_config(configs):
    for name, config in configs.items():
        print('====={}====='.format(name))
        _print_config(config)
        print('=' * (len(name) + 10))
        print()


def _print_config(config, indentation=''):
    for key, value in config.items():
        if isinstance(value, dict):
            print('{}{}:'.format(indentation, key))
            _print_config(value, indentation + '    ')
        else:
            print('{}{}: {}'.format(indentation, key, value))


def print_dataset_info(datasets):
    train_dataset, test_dataset, val_dataset = datasets
    print('=========================')
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    print('=========================')


# unnormalize image for visualization
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# convert labels to onehot
def one_hot(labels, num_classes, device, dtype):
    y = torch.eye(num_classes, device=device, dtype=dtype)
    return y[labels]

# convert type of target according to criterion
def select_target_type(y, criterion):
    if criterion in ['cross_entropy', 'kappa_loss']:
        y = y.long()
    elif criterion in ['mean_square_error', 'mean_absolute_error', 'smooth_L1']:
        y = y.float()
    elif criterion in ['focal_loss']:
        y = y.to(dtype=torch.int64)
    else:
        raise NotImplementedError('Not implemented criterion.')
    return y


# convert output dimension of network according to criterion
def select_out_features(num_classes, criterion):
    out_features = num_classes
    if criterion in regression_loss:
        out_features = 1
    return out_features


def matplotlib_roccurve(fpr_tpr_tuples, labels, points=None, point_labels=None):    
    """helper function to plot a roc curve
        fpr_tpr_tuples: list of tuples [(fpr, tpr), (fpr, tpr)]
    """
    
    if not len(fpr_tpr_tuples) == len(labels):
        raise ValueError('both inputs must have same length.')
        
    if points is not None:
        if not len(points) == len(point_labels):
            raise ValueError('both inputs must have same length.')
        
        
    fig, ax = plt.subplots()
    
    for ii in range(len(fpr_tpr_tuples)):
        plt.plot(fpr_tpr_tuples[ii][0], fpr_tpr_tuples[ii][1], label=labels[ii])
        
    if points is not None:     
        for ii in range(len(points)):
            plt.plot(points[ii][0], points[ii][1], 'o', label=point_labels[ii])
            
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    ax.set_aspect('equal')

    return fig

def matplotlib_prec_recall_curve(precison_recall_tuples, labels):    
    """helper function to plot a roc curve
        fpr_tpr_tuples: list of tuples [(fpr, tpr), (fpr, tpr)]
    """
    
    if not len(precison_recall_tuples) == len(labels):
        raise ValueError('both inputs must have same length.')        
        
    fig, ax = plt.subplots()
    
    for ii in range(len(precison_recall_tuples)):
        plt.plot(precison_recall_tuples[ii][0], precison_recall_tuples[ii][1], label=labels[ii])
            
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    ax.set_aspect('equal')

    return fig

def plot_conf_matrix(cm):
    """  combine the confusion matrix with the approproate labels to make it easier to visualize """

    column, indices = [0,1], [0,1]
    table  = pd.DataFrame(cm, columns=column, index=indices)
    return sns.heatmap(table, annot=True, fmt='d', cmap='viridis')