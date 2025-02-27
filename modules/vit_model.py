
import timm
import torch
import torch.nn as nn
from torchvision import models

from transformers import ViTForImageClassification
from timm.models.swin_transformer import SwinTransformer
from transformers import SwinConfig, AutoImageProcessor, SwinForImageClassification

def ViT_Tim_Model(num_classes):
    model_name = "vit_base_patch16_384"
    model = timm.create_model(model_name, pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def SwinTimmModel(num_classes):
    print('Swin Timm model')
    #model_name = "swin_base_patch4_window12_384"
    model_name = "swin_large_patch4_window12_384"
    model = timm.create_model(model_name, pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)

    return model