
import torch
import torch.nn as nn

class SSDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        loss_loc = self.loc_loss(loc_preds, loc_targets)
        loss_cls = self.cls_loss(cls_preds, cls_targets)
        return loss_loc + loss_cls
