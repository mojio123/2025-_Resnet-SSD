import torch
import torch.nn as nn
import torch.optim as optim

# 定义FCNN模型
class FusionFCNN(nn.Module):
    def __init__(self):
        super(FusionFCNN, self).__init__()

        # 高光谱图像处理网络
        self.hyperspectral_conv1 = nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, padding=1)
        self.hyperspectral_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.hyperspectral_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.hyperspectral_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RGB图像处理网络
        self.rgb_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.rgb_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rgb_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.rgb_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 融合部分
        self.fusion_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        # 输出层
        self.output_conv = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1)  # 输出RGB图像

    def forward(self, hyperspectral, rgb):
        # 高光谱图像前向传播
        x_hyper = torch.relu(self.hyperspectral_conv1(hyperspectral))
        x_hyper = self.hyperspectral_pool1(x_hyper)

        x_hyper = torch.relu(self.hyperspectral_conv2(x_hyper))
        x_hyper = self.hyperspectral_pool2(x_hyper)

        # RGB图像前向传播
        x_rgb = torch.relu(self.rgb_conv1(rgb))
        x_rgb = self.rgb_pool1(x_rgb)

        x_rgb = torch.relu(self.rgb_conv2(x_rgb))
        x_rgb = self.rgb_pool2(x_rgb)

        # 融合特征
        x_fused = torch.cat([x_hyper, x_rgb], dim=1)  # 沿通道维度拼接

        # 融合后的卷积
        x_fused = torch.relu(self.fusion_conv1(x_fused))
        x_fused = torch.relu(self.fusion_conv2(x_fused))

        # 输出
        output = self.output_conv(x_fused)
        return output

# 创建模型实例
model = FusionFCNN()


print(model)

