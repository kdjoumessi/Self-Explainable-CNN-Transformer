import torch
import torch.nn as nn
import bagnets.pytorchnet as bpy

from einops import rearrange
from torchvision import models
from collections import OrderedDict
from .drsa_main_compoments import DRSATransformerBlock, DRSAConvTransformerBlock  

class DRSA_model(nn.Module):
    def __init__(self, cfg):
        super(DRSA_model, self).__init__()

        layers = {} 
        self.cfg = cfg  
        num_head = self.cfg.drsa.num_head     
        num_classes  = cfg.data.num_classes
        self.head_size = self.cfg.drsa.head_size
        self.res = 'resnet' in cfg.train.network
        self.reduct_fact = self.cfg.drsa.reduction_factor

        if self.res:
            n = 16
            layer_0_names = ['conv1', 'bn1', 'relu', 'maxpool']
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT') 
        else:
            n=60
            layer_0_names = ['conv1', 'conv2', 'bn1', 'relu']
            model = bpy.bagnet33(pretrained=True)
        in_ftrs = model.fc.in_features

        # extract layers while preserving the names
        for name, layer in model.named_children():
            layers[name] = layer

        layers_0 = {}
        for name in layer_0_names:
            layers_0[name] = layers[name]
            
        # convert into OrderectDict and create the sequential model from the OrderedDict
        layers_0 = OrderedDict(layers_0)
        self.layer0 = nn.Sequential(layers_0)
        
        self.layer1 = layers['layer1'] # already sequential
        self.layer2 = layers['layer2']
        self.layer3 = layers['layer3']
        self.layer4 = layers['layer4']

        if cfg.drsa.conv_drsa:
            mhsa = DRSAConvTransformerBlock(self.cfg, dim=n*n, channels=in_ftrs)
        else:
            mhsa = DRSATransformerBlock(self.cfg, dim=n*n, channels=in_ftrs)
        self.mhsa = mhsa

        if cfg.train.conv_cls:
            self.classifier = nn.Conv2d(model.fc.in_features, num_classes, kernel_size=(1, 1), stride=1)
            self.avgpool = nn.AvgPool2d(kernel_size=(n,n), stride=(1,1), padding=0)
        else:    
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            self.classifier = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x): 
        x0 = self.layer0(x)   #                   => (bs, 64,   510, 510)
        x1 = self.layer1(x0)  #                   => (bs, 256,  254, 254)
        x2 = self.layer2(x1)  #                   => (bs, 512,  126, 126)
        x3 = self.layer3(x2)  #                   => (bs, 1024, 62,  62)
        xhigh = self.layer4(x3)  #                => (bs, 2028, 60,  60)

        b, c, h, w = xhigh.shape   
        xhigh_att = self.mhsa(xhigh, res=0)  # res=0 ~ high-resolution 
        xlow_att = None
        res = 0

        if self.cfg.drsa.with_drsa:
            xlow = self.mhsa.downsample(xhigh)   # downsample the input to low-resolution
            xlow_att = self.mhsa(xlow, res=1)
            res = 1
             
        att_weight = self.mhsa.final_attention(xlow_att, xhigh_att, xhigh, res)  # (bs, C, H, W) ~ (bs, 2048, 60, 60)
        b, nc, h, w = att_weight.shape

        if self.cfg.train.conv_cls:
            x_heatmap = self.classifier(att_weight)        # (bs, nclasses, H, W)
            out = self.avgpool(x_heatmap)                  # (bs, nclasse, 1, 1)
            out = out.view(out.shape[0], -1)               # (bs, nclasse)
        else:
            x = self.avgpool(att_weight)    # (bs, C, 1, 1)
            x = torch.flatten(x, 1)         # (bs, C)
            out = self.classifier(x)        # (bs, nclasse)
            x_heatmap = torch.zeros((b, nc, h, w), device='cuda')   

        return out, x_heatmap, att_weight    