'''
客户给的文件比较杂，不敢乱动
通过init提供模型调用接口
目前除了Swin都调通了，但是其他模型的预测效果都算不上好
一般蛋白质含量17%-27%
一般油脂含量34%-50%
'''
from typing import Literal
# 导入模型全部改成options一致的名字，方便调用
from .MPViT import MPViT # 导入MPViT模型
from .Swin import SwinV2 as Swin # 导入Swin模型 多封装一层以适配模型
from .VanillaNet import VanillaNet
from .FasterNet import model as FasterNet # 导入FasterNet模型
from .ResNet18 import ResNet18 as ResNet # 导入ResNet18模型
from .EfficientNet import EfficientNet # 导入EfficientNet模型
from .yolo import yolo_model as YOLO # 导入YOLO模型

REGRESS_MODEL_OPTIONS = Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet'] # 回归模型选项
DETECT_MODEL_OPTIONS = Literal['YOLO'] # 检测模型选项

# 原本是从客户给的文件里导入，但是每个文件的调用方法不同，就全部从新封装了统一的方法
# from .build_mpvit import MPViT # 导入MPViT模型
# from .build_swinv2 import SwinTransformerV2 as Swin # 导入SwinTransformerV2模型
# from .build_vanillanet import VanillaNet # 导入VanillaNet模型
# from .model_zoo import FasterNet # 导入FasterNet模型
# from .model_zoo import Efficient_net as EfficientNet # 导入EfficientNet模型

