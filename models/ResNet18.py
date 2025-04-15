'''
基于torch的ResNet18再封装以适应weights\ResNet18_best.pt
主要差别是最后输出的时候多了一个全连接层fc1，将输出转换为量输出，推测是应该是蛋白质和油脂的含量
'''

import torch
from torch import nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes = 2, device = 'cpu'):
        self.device = device
        super(ResNet18, self).__init__()
        # 加载预训练的ResNet18模型
        self.ResNet = resnet18(pretrained=False)  # 这里设置为False，因为我们将加载本地的预训练模型
        # 增加一层全连接层，用于将ResNet18的输出转换为数量输出
        self.fc1 = nn.Linear(self.ResNet.fc.out_features, num_classes)  # 假设我们要预测的是数量，所以输出维度为2

    def forward(self, x):
        # 前向传播，先通过ResNet18提取特征，然后通过全连接层得到数量输出
        x = self.ResNet(x)  # 提取特征
        out = self.fc1(x)  # 转换为数量输出
        out = out[0]
        out[1] = out[1] * (29.1-17.4) / 100 + 17.4
        out[0] = out[0] * (50.5 - 32.5) / 100 + 32.5
        return out

    def load_model_weight(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location=torch.device(self.device)), strict=True)
        return self.state_dict()
    
if __name__ == '__main__':
    # # 示例用法
    # model = ResNet18(num_classes=2)  # 创建模型实例，假设我们要预测的是数量，所以输出维度为2
    # # 加载本地的预训练模型
    # state_dict = torch.load('weights/ResNet18_best.pt', map_location='cpu')  # 加载本地的预训练模型

    # model.load_state_dict(state_dict)  # 加载本地的预训练模型
    # # 加载本地的预训练模型
    model = resnet18(pretrained=False)  # 加载本地的预训练模型
    state_dict = torch.load('weights/ResNet18_best.pt', map_location='cpu')  # 加载本地的预训练模型
    std_dict = model.state_dict()  # 获取模型的状态字典
    model.load_state_dict(state_dict, strict=False)  # 加载状态字典
    with open('resnet18_model_structure.txt', 'w') as f:
        f.write(str(model))
    with open('resnet18_model_state_dict.txt', 'w') as f:
        for key, value in std_dict.items():
            f.write(f"{key}: {value.shape}\n")

    mk,uk = model.load_state_dict(state_dict, strict=False)
    print('missing_keys:', mk)
    print('unexpected_keys:', uk)