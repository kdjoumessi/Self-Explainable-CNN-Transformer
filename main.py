import os
import sys
import time
import torch
import random
import numpy as np

from utils.func import *
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model

from torch.utils.tensorboard import SummaryWriter

def main():
    # load conf and paths files
    args = parse_config()
    cfg, cfg_paths = load_config(args)   
    cfg = data_path(cfg, cfg_paths)

    if cfg.train.vit:
        cfg.train.network = 'vit'
        cfg.train.conv_cls = False
        cfg.train.drsa = False
        cfg.data.input_size = 384

    if cfg.train.res_baseline:
        cfg.train.train_with_att = False
        cfg.train.conv_cls = False
        cfg.train.vit = False
        cfg.train.drsa = False

    if cfg.train.bag_baseline:
        cfg.train.train_with_att = False
        cfg.train.conv_cls = True
        cfg.train.vit = False
        cfg.train.drsa = False

    if not cfg.drsa.conv_drsa:
        cfg.train.conv_cls = False
        cfg.train.train_with_att = True

    if cfg.train.network == 'resnet50':
        cfg.drsa.lr_dim = 8
        cfg.drsa.head_size = 8
        cfg.drsa.ld_head_size = 8


    # Test
    if cfg.base.test:
        print('########## test')
        cfg.base.test = True
        cfg.base.sample = 10
        cfg.train.epochs = 4
        #cfg.train.n_head = 5
        cfg.train.batch_size = 1
     
    # create folder
    cfg.dset.save_path = load_save_paths(cfg)
    save_path = cfg.dset.save_path 

    cfg.base.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = SummaryWriter(save_path)    

    folders = ['configs', 'data', 'modules', 'utils']
    copy_file(folders, save_path)
    
    # print configuration
    if args.print_config:
        print_config({
            'BASE CONFIG': cfg.base,
            'DATA CONFIG': cfg.data,
            'TRAIN CONFIG': cfg.train
        })
    else:
        print_msg('LOADING CONFIG FILE: {}'.format(args.config))

    since = time.time()
    set_random_seed(cfg.base.random_seed)  

    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)  
    model = generate_model(cfg)

    estimator = Estimator(cfg)
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
        logger=logger
    )

    ################## test ############
    ## Manual test 
    name = 'best_validation_weights'
    model_list = ['acc', 'kappa'] 
    model_name = ['Accuracy', 'Kappa'] 

    for i in range(len(model_list)):
        print('========================================')
        print(f'This is the performance of the final model base on the best {model_name[i]}')
        checkpoint = os.path.join(save_path, f'{name}_{model_list[i]}.pt')

        evaluate(cfg, model, checkpoint, val_dataset, estimator, type_ds='validation')
    
        print('')
        evaluate(cfg, model, checkpoint, test_dataset, estimator, type_ds='test')
        print('')

    time_elapsed = time.time() - since
    print('Training and evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()
