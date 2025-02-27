import os
import yaml 
import torch
import numpy as np
import seaborn as sns

from tqdm import tqdm
from PIL import Image
from munch import munchify
from torchvision import transforms

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

############## General ###############
######################################

#------------------------
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

#-----------------------------------------
def load_image(cfg, path, img_name):
    '''
    return the corresponding image (tensor and numpy): => (batch_size, C, H, W).

        Parameters:
            - path (str): image path location
        
        Returns:
            - PIL normalize (in [0, 1]) with shape (b, C,H,W)  
            - np image unormalize image
    '''
    pil_img = Image.open(os.path.join(path, img_name)) 
    
    normalization = [
        transforms.Resize((cfg.data.input_size)), 
        transforms.ToTensor(),
        transforms.CenterCrop(cfg.data.input_size),
        transforms.Normalize(cfg.data.mean, cfg.data.std)]
    
    test_preprocess_norm = transforms.Compose(normalization)
    
    ts_img = torch.unsqueeze(test_preprocess_norm(pil_img), dim=0)   
    np_img = np.array(pil_img)                                
    
    return ts_img, np_img

#-----------------------------------------
def get_inference(cfg, network, df, f_path, res_model=True, model='res', bar=True):
    ''' 
       description: inference on the test set
    '''
    
    yconf, ypred = [], []
    activations, attentions = {}, {}
    
    dat = df.copy()
    filenames   = df['filename'].tolist()

    iterate = tqdm(filenames) if bar else filenames
    
    for fname in iterate:
        ts_img, np_img = load_image(cfg, f_path, fname)
        ts_img = ts_img.to('cuda')
        
        if res_model:
            pred = network(ts_img)
        else:
            pred, act, att = network(ts_img)
            #act = act.detach().cpu().numpy()
            #att = att.detach().cpu().numpy()
            
            #activations[fname] = act
            #attentions[fname] = att

        y_prob = torch.nn.functional.softmax(pred.detach().cpu(), dim=1)
        val, idx = torch.topk(y_prob, k=1)
        
        ypred.append(idx.item())
        yconf.append(round(val.item(), 3))

    dat[f'{model}_conf'] =  yconf
    dat[f'{model}_pred'] = ypred
    
    return dat, activations, attentions

############## Plot ###############
######################################

#-----------------------------------------
def plot_img_heat_att(title, imgs, size, acts=False, prob=None, txt_kwargs=None):
    fig = plt.figure(figsize=size, layout='constrained') # 
    j = 0
    n =  len(acts) if acts else len(imgs)
    nrow = 2 if acts else 1
    
    for i, img_ in enumerate(imgs):
        j += 1
        ax = fig.add_subplot(nrow, n, j)
        img = ax.imshow(img_, cmap='viridis')
        ax.set_title(title[i], loc="center", **txt_kwargs)
        ax.axis('off')
    divider = make_axes_locatable(ax)
    
    j = 5
    if acts:
        for i in range(len(acts)):
            j += 1
            ax = fig.add_subplot(nrow, n, j)
            img = ax.imshow(acts[i], cmap='viridis')
            ax.set_title(prob[i], loc="center", **txt_kwargs)
            ax.axis('off')
    cbar = fig.colorbar(img, ax=ax, location='right', shrink=0.85)

#-----------------------------------------
def plot_cm(df, model_name, ncol=3, scale=1.4, fs=12, log=True, size=(5,5)):
    fig = plt.figure(figsize=size, layout='constrained')
    sns.set(font_scale=scale)

    for idx, (m_name, cname) in enumerate(model_name.items()):
        class_acc_msg = 'Class accuracy: '
        acc = round(accuracy_score(df[cname], df['level']), 3)
        kappa = round(cohen_kappa_score(df['level'], df[cname], weights='quadratic'), 3)
        
        if log:
            for i in range(5):
                n_total = len(df[df.level==i])
                n_good_pred = len(df[(df.level==i) & ((df[cname]==i))])
                c_acc = round(n_good_pred / n_total, 3)
                class_acc_msg += f'class {i}: {c_acc} \t'
    
            print(f'{m_name} model, \t Accuracy: {acc}, \t Kappa: {kappa}')
            print(f'{class_acc_msg} \n')

        cm  = confusion_matrix(df['level'], df[cname])
        ax = fig.add_subplot(1, ncol, idx+1)
        sns.heatmap(cm, annot=True, fmt='.0f', annot_kws={"size": fs}, cbar=False, ax=ax)
        ax.set_title(m_name)