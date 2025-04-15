from .model_zoo import mpvit_base
import torch
import torch.nn as nn

class Cls_head(nn.Module):
    """a linear layer for classification."""
    def __init__(self, embed_dim, num_classes):
        """initialization"""
        super().__init__()

        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """foward function"""
        # (B, C, H, W) -> (B, C, 1)

        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        # Shape : [B, C]
        out = self.cls(x)
        return out
    
class MPViT(nn.Module):
    def __init__(self, weight = None, number_of_classes = 2, device = 'cpu'):
        self.device = device
        super(MPViT, self).__init__()
        self.model = mpvit_base(num_classes=1000)
        # load_pretrained(self.model, weight)
        self.model.cls_head = Cls_head(480, number_of_classes)
    
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
        self.to(self.device)
        state_dict = torch.load(weight_path, map_location=torch.device(self.device))
        self.load_state_dict(state_dict, strict=True) # 这里有点不理解的地方，调用model变量的load_state_dict函数会报错，可能是我直接复制的模型代码而不是封装原先的模型导致的所以，相当于再原先的模型上改
        return self.state_dict()