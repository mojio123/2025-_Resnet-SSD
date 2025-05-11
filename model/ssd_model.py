
import torch
import torch.nn as nn
from model.resnet_backbone import ResNetBackbone
from model.ssd_head import SSDHead

class SSDResNet(nn.Module):
    def __init__(self, num_classes):
        super(SSDResNet, self).__init__()
        self.backbone = ResNetBackbone()
        self.head = SSDHead(num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        locs, confs = self.head(feats)
        return locs, confs
