from .model_zoo import FasterNet, init_weights

import torch
from torch import nn
'''
Val_y_true[i][1] = Val_y_true[i][1] * (29.1-17.4) / 100 + 17.4
Val_y_true[i][0] = Val_y_true[i][0] * (50.5 - 32.5) / 100 + 32.5

'''
class model(nn.Module):
    def __init__(self, num_classes = 2, device = 'cpu'):
        self.device = device
        super(model, self).__init__()
        self.model = FasterNet(num_classes=num_classes)
    
    def forward(self, x):
        out = self.model(x)
        out = out[0]
        out[1] = out[1] * (29.1-17.4) / 100 + 17.4
        out[0] = out[0] * (50.5 - 32.5) / 100 + 32.5
        return out
    
    def load_model_weight(self, weight_path):
        # 初始化权重报错，但是日志没输出，还没搞明白，我重新写一个权重加载
        # init_weights(self.model, pretrained=weight_path)
        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict,strict=True)
        return self.model.state_dict()
    
if __name__ == "__main__":
    model = model()
    state_dict = torch.load('weights\FasterNet.pt')
    model.load_state_dict(state_dict)
    print(model)