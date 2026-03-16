'''
这里定义在fastapi中用到的数据类
如果有需要可以方便扩展，目前只有一种数据类型应该是
'''
from typing import Union, Literal
from pydantic import BaseModel
from datetime import datetime

class TaskModel(BaseModel):
    '''
    用于接收任务json的数据类
    '''
    image_url: str  # 统一使用image_url字段名
    model: Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet']  # 统一使用model字段名

class DetectAndEvalModel(BaseModel):
    '''
    用于接收检测和评估任务的数据类
    '''
    image_url: str  # 统一使用image_url字段名
    model: Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet'] = 'FasterNet'  # 统一使用model字段名
    conf_threshold: Union[float, None] = None  # 检测置信度阈值，None表示使用默认值
    iou_threshold: Union[float, None] = None   # IoU阈值，None表示使用默认值

class RipenessModel(BaseModel):
    '''
    用于接收成熟度分类任务的数据类
    '''
    image_url: str  # 统一使用image_url字段名

class DetectAndEvalModel3(BaseModel):
    '''
    用于接收 V3 检测和评估任务的数据类（不需要 model 参数）
    '''
    image_url: str  # 统一使用image_url字段名
    conf_threshold: Union[float, None] = None  # 检测置信度阈值，None表示使用默认值
    iou_threshold: Union[float, None] = None   # IoU阈值，None表示使用默认值

# AI-GENERATED-START
class RapeseedPadimModel(BaseModel):
    '''
    用于接收 PaDiM 菜籽判别任务的数据类
    '''
    image_url: str
    threshold_buffer_a: Union[int, None] = None
    threshold_buffer_b: Union[int, None] = None
# AI-GENERATED-END
