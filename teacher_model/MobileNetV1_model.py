import torch
from torch import nn
from torchsummary import summary

#定义ReLU6激活函数
class ReLU6(nn.Module):
    def __init__(self):
        super(ReLU6, self).__init__()
    
    def forward(self, x):
        return torch.clamp(x, 0, 6)

# 定义深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积 - 逐通道卷积
        self.depthwise = nn.Conv2d(in_channels=in_channels, 
                                   out_channels=in_channels, 
                                   kernel_size=3, 
                                   stride=stride, 
                                   padding=1, 
                                   groups=in_channels, # 关键参数：groups=in_channels实现深度卷积,每个通道独立卷积
                                   bias=False)         #若使用批归一化（BatchNorm），通常设置 bias=False ，因为BN层已包含偏置调整
        # 逐点卷积 - 1x1卷积融合特征
        self.pointwise = nn.Conv2d(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=1, 
                                   stride=1, 
                                   padding=0, 
                                   bias=False)  
        # 批量归一化和ReLU激活
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = ReLU6()
    
    def forward(self, x):
        # 深度卷积 + BN + ReLU
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 逐点卷积 + BN + ReLU
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

# 定义MobileNetV1模型
class MobileNetV1(nn.Module):
    def __init__(self, DepthwiseSeparableConv):
        super(MobileNetV1, self).__init__()
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 深度可分离卷积块列表，遵循MobileNetV1论文架构
        self.depthwise_blocks = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),      # 32->64, stride=1
            DepthwiseSeparableConv(64, 128, stride=2),     # 64->128, stride=2
            DepthwiseSeparableConv(128, 128, stride=1),    # 128->128, stride=1
            DepthwiseSeparableConv(128, 256, stride=2),    # 128->256, stride=2
            DepthwiseSeparableConv(256, 256, stride=1),    # 256->256, stride=1
            DepthwiseSeparableConv(256, 512, stride=2),    # 256->512, stride=2
            
            # 5个连续的512通道深度可分离卷积，stride=1
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            
            DepthwiseSeparableConv(512, 1024, stride=2),   # 512->1024, stride=2
            DepthwiseSeparableConv(1024, 1024, stride=1)   # 1024->1024, stride=1
        )
        
        # 平均池化和全连接层（CIFAR10有10个类别）
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    
    def forward(self, x):
        # 初始卷积
        x = self.initial_conv(x)
        # 深度可分离卷积序列
        x = self.depthwise_blocks(x)
        # 池化、展平、全连接
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # 设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    
    # 创建模型实例并移动到指定设备
    model = MobileNetV1(DepthwiseSeparableConv).to(device)
    
    # 打印模型结构摘要，输入尺寸为(3, 224, 224)（与ResNet18保持一致）
    print(summary(model, (3, 224, 224)))