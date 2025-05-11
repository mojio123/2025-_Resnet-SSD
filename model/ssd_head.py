
import torch.nn as nn
class SSDHead(nn.Module):
    def __init__(self, num_classes, num_boxes=[4, 6, 6, 6, 4, 4]):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes
        self.loc = nn.ModuleList()
        self.cls = nn.ModuleList()


        in_channels = [2048, 1024, 512, 512, 256, 256]
        for i, channels in enumerate(in_channels):
            self.loc.append(nn.Conv2d(channels, num_boxes[i]*4, kernel_size=3, padding=1))
            self.cls.append(nn.Conv2d(channels, num_boxes[i]*num_classes, kernel_size=3, padding=1))

    def forward(self, features):
        locs = []
        confs = []
        for i, f in enumerate(features):
            locs.append(self.loc[i](f).permute(0, 2, 3, 1).contiguous())
            confs.append(self.cls[i](f).permute(0, 2, 3, 1).contiguous())
        return locs, confs
