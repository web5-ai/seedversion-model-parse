from .model_zoo import vanillanet_13
import torch
import torch.nn as nn

class VanillaNet(nn.Module):
    def __init__(self, weight, number_of_classes = 2):
        super(VanillaNet, self).__init__()
        self.model = vanillanet_13(num_classes=1000)
        
        self.model.load_state_dict(torch.load(weight)['model'], strict=False)
        # load_pretrained(self.model, weight)
        
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
        return self.func_transition(head_out)
    
if __name__ == "__main__":
    model = VanillaNet(weight=None)
    print(model)