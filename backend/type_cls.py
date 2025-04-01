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
    timestamp: datetime 
    usr_id:str
    img_src:str
    model_name:Literal['ResNet','VGG','FasterNet']
    model_path:Union[str, None] =None