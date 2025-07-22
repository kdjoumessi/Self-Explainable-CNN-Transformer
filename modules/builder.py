import copy
import torch
import torch.nn as nn
import bagnets.pytorchnet

from torchvision import models
from .bagnet import SparseBagnet
from .drsa_models import DRSA_model
from .vit_model import SwinTimmModel, ViT_Tim_Model
from utils.func import print_msg, select_out_features

def generate_model(cfg):
    out_features = select_out_features(cfg.data.num_classes, cfg.train.criterion)

    model = build_model(cfg, out_features)

    if cfg.base.device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(cfg.base.device)
    return model


def build_model(cfg, num_classes):
    network = cfg.train.network
    
    pretrained = cfg.train.pretrained

    if cfg.train.drsa:
        model = DRSA_model(cfg)
    elif cfg.train.vit:
        model = ViT_Tim_Model(num_classes)
    else:
        if cfg.train.train_with_att:
            model = MHSA_model(cfg, num_classes, pretrained)
        elif network == 'resnet50':
            model = BUILDER[network](weights='ResNet50_Weights.DEFAULT')
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'bagnet' in network: 
            model = BUILDER[network](pretrained=pretrained)
            model = SparseBagnet(model, num_classes)
        else:
            raise NotImplementedError('Not implemented network.')

    return model


## Sparse BagNet
class SparseBagnet(nn.Module):
    def __init__(self, model, num_classes):
        super(SparseBagnet, self).__init__()

        last_block = list(model.layer4.children())[-1] 
        num_channels = last_block.conv3.out_channels

        backbone = list(model.children())[:-2]
        self.backbone = nn.Sequential(*backbone)
        
        # classification layer: conv2d instead of FCL
        self.classifier = nn.Conv2d(num_channels, num_classes, kernel_size=(1,1), stride=1)
        self.clf_avgpool = nn.AvgPool2d(kernel_size=(1,1), stride=(1,1), padding=0)

    def forward(self, x):        
        x = self.backbone(x)                # (bs, c, h, w) 
        activation = self.classifier(x)     # (bs, n_class, h, w) 
        bs, c, h, w = x.shape               # (bs, n_class, h, w) 
          
        avgpool = nn.AvgPool2d(kernel_size=(h, w), stride=(1,1), padding=0)             
        out = avgpool(activation)           # (bs, n_class, 1, 1) 
        out = out.view(out.shape[0], -1)    # (bs, n_class)

        att_weight = torch.zeros((bs, c, h, w), device='cuda')
        
        return out, activation, att_weight


BUILDER = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'wide_resnet50': models.wide_resnet50_2,
    'wide_resnet101': models.wide_resnet101_2,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    'mobilenet': models.mobilenet_v2,
    'squeezenet': models.squeezenet1_1,
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,
    'bagnet33': bagnets.pytorchnet.bagnet33,
    'bagnet17': bagnets.pytorchnet.bagnet17,
    'bagnet9': bagnets.pytorchnet.bagnet9
}     