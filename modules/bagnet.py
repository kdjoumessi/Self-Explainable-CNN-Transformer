
import torch
import torch.nn as nn

class SparseBagnet(nn.Module):
    def __init__(self, model, num_classes):
        super(SparseBagnet, self).__init__()

        last_block = list(model.layer4.children())[-1] 
        num_channels = last_block.conv3.out_channels

        backbone = list(model.children())[:-2]
        self.backbone = nn.Sequential(*backbone)
        
        # classification layer: conv2d instead of FCL
        self.classifier = nn.Conv2d(num_channels, num_classes, kernel_size=(1,1), stride=1)
        # 24 ~ 224, 60~ 512
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