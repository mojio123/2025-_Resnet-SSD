
import torch
import torch.nn as nn
import torchvision.models as models
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.extra = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1),  # 2048 -> 1024
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),  # 1024 -> 512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),  # 512 -> 512
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # 512 -> 256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),  # 256 -> 256
            nn.ReLU()
        )

    def forward(self, x):
        x = self.base(x)
        sources = [x]
        for layer in self.extra:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                sources.append(x)
        return sources[:6]
