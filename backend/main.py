from typing import Union, Literal
from model_api import ModelAPI
from fastapi import FastAPI
from fastapi.responses import FileResponse
import numpy as np
import json
app = FastAPI()

# 初始化模型API，加载默认模型
model_api = ModelAPI()

@app.get("/")
def root():
    return {"Hello": "World"}

@app.get("/image")
def get_image():
    '''
    测试接口，用来给predict返回图片，异步下不可用会阻塞
    '''

    return FileResponse('tests/test_images/image_custom.png')

@app.post("/predict")
async def predict(image_url: str, model_name: Literal['ResNet','VGG','FasterNet'], model_path = None)->dict:
    '''
    预测接口，接收图像文件和模型名称，返回预测结果。这里没用异步等待，因为选用的路由是项目内的，会阻塞
    Args:
        image_url: 图像文件的URL或路径
        model_name: 模型名称'
        model_path: 模型文件的路径，默认为None
    Returns:
        预测结果字典
    '''
    # 读取上传的图像文件
    report = model_api.eval_image(image_url, model_name, model_path) # 得到字典
    # 将字典转为JSON字符串，其中的numpy数组会自动转换为列表
    for key, value in report.items():
        if isinstance(value, np.ndarray):
            report[key] = value.tolist()
    return report

# if __name__ == "__main__":
#     import uvicorn
#     # 访问127.0.0.1:8000/docs查看文档
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level='debug')