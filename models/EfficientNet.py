from .model_zoo import Efficient_net

# 再封装
import torch
from torch import nn

class EfficientNet(nn.Module):
    def __init__(self, number_classes = 2, device = 'cpu'):
        self.device = device
        super(EfficientNet, self).__init__()
        self.model = Efficient_net(num_classes=number_classes)

    def forward(self, x):
        out = self.model(x)
        out = out[0]
        out[1] = out[1] * (29.1-17.4) / 100 + 17.4
        out[0] = out[0] * (50.5 - 32.5) / 100 + 32.5
        return out
    
    def load_model_weight(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device(self.device)), strict=True)
        return self.model.state_dict()