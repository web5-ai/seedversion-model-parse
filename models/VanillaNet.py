'''
根据build_vanillanet.py中的代码，封装一个类
'''

from .model_zoo import vanillanet_13
import torch
import torch.nn as nn

class VanillaNet(nn.Module):
    def __init__(self, weight = None, number_of_classes = 2, device = 'cpu'):
        self.device = device
        super(VanillaNet, self).__init__()
        self.model = vanillanet_13(num_classes=1000)
        
        self.model.cls1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0),
            nn.Conv2d(1024*4, number_of_classes, 1),
            # nn.BatchNorm2d(num_classes, eps=1e-6),
        )
        self.model.cls2 = nn.Sequential(
            nn.Conv2d(number_of_classes, number_of_classes, 1)
        )
    
    def func_transition(self,outputs):
        # outputs = outputs.view(-1)

        outputs[:,0] = (torch.sigmoid(outputs[:,0]) * 100)
        outputs[:,1] = (torch.sigmoid(outputs[:,1]) * 100)
        return outputs[:, :2]
        
    def forward(self, x):
        head_out = self.model(x)

        out = self.func_transition(head_out)
        out = out[0]
        out[1] = out[1] * (29.1-17.4) / 100 + 17.4
        out[0] = out[0] * (50.5 - 32.5) / 100 + 32.5
        return out
    
    def load_model_weight(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location=torch.device(self.device)), strict=True)
        return self.state_dict()
    
if __name__ == "__main__":
    model = VanillaNet(weight=None)
    print(model)