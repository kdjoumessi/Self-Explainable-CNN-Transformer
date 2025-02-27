import os
import cv2
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

class FundusDataset(Dataset):
    def __init__(self, cfg, train=True, test=False, transform=None):
        self.img_size = (cfg.data.input_size, cfg.data.input_size)
        self.transform = transform
        self.image_path = os.path.join(cfg.dset.root, cfg.dset.data_dir)
        self.str_label = cfg.data.target
        self.n_classes = cfg.data.num_classes

        if not test: 
            if train:
                self.df = pd.read_csv(os.path.join(cfg.dset.train_csv))
            else:
                self.df = pd.read_csv(os.path.join(cfg.dset.val_csv))
        else:
            self.df = pd.read_csv(os.path.join(cfg.dset.test_csv))

        self.filenames = self.df[cfg.data.fname]
        self.labels = self.df[self.str_label]
        self.classes = sorted(list(set(self.targets)))
        
        if cfg.base.test:            
            n=cfg.base.sample
            self.filenames = self.filenames[:n]
            self.labels = self.labels[:n]
        

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = os.path.join(self.image_path, self.filenames[idx])        
        
        image = Image.open(filename) # generate PIL image

        if self.transform:
            image = self.transform(image)
            
        label = self.labels.iloc[idx]

        return image, label
    
    def balanced_weights(self):

        #class_prop = [0.3, 0.7]    
        # class_weights = (1./len(class_weights)) / class_weights
        weights = [0] * len(self)

        for idx, val in enumerate(getattr(self.df, self.str_label)):
                weights[idx] = 1 / self.class_proportions[val]
        return weights

    @property
    def class_proportions(self):
        y = self.targets.view(-1, 1)
        targets_onehot = (y == torch.arange(self.n_classes).reshape(1, self.n_classes)).float()
        proportions = torch.div(torch.sum(targets_onehot, dim=0) , targets_onehot.shape[0])
        return proportions

    @property
    def targets(self):
        return torch.tensor(self.df[self.str_label].values)
