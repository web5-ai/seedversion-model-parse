from .model_zoo import Efficient_net,FasterNet,init_weights,SwinTransformerV2,mpvit_base,vanillanet_13_x1_5,vanillanet_5,vanillanet_13
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
    def __init__(self, weight = None, number_of_classes = 2):
        super(MPViT, self).__init__()
        self.model = mpvit_base(num_classes=1000)
        
        self.model.load_state_dict(torch.load(weight)['model'], strict=False)
        # load_pretrained(self.model, weight)
        self.model.cls_head = Cls_head(480, number_of_classes)
    
    def func_transition(self,outputs):
        # outputs = outputs.view(-1)

        outputs[:,0] = (torch.sigmoid(outputs[:,0]) * 100)
        outputs[:,1] = (torch.sigmoid(outputs[:,1]) * 100)
        return outputs[:, :2]
        
    def forward(self, x):
        head_out = self.model(x)
        return self.func_transition(head_out)