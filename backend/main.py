from fastapi import FastAPI
from fastapi.responses import FileResponse

from type_cls import TaskModel
from tools import *
from model_api import ModelAPI

import datetime

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
async def predict(task_info:TaskModel)->dict:
    '''
    预测接口，接收图像文件和模型名称，返回预测结果。
    Args:
        task_info:TaskModel
            timestamp: 任务上传时间（采样时间）需要约定时间格式便于转换存储
            usr_id: 用户id
            image_src: 图像文件的URL或路径
            model_name: Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet']

    Returns:
        预测结果字典，fastapi自动转为json
    '''
    # 还没有约定好时间传递方式，不好转换，这里我先自定一个iso格式转换
    timestamp = task_info.timestamp.strftime("%Y_%m_%d-%H_%M_%S")
    # 读取上传的图像文件

    img = get_img(task_info.img_src)
    # 模型预测
    s = datetime.datetime.now()
    evals = model_api.eval_image(img, task_info.model_name) # 得到字典
    e = datetime.datetime.now()
    evals['time_delta'] = (e-s).total_seconds()
    # 保存任务到数据库，返回任务id，用于查询任务状态和结果
    task_data = {
        'timestamp': timestamp,
        'usr_id': task_info.usr_id,
        'image': img,
        'model': task_info.model_name,
        'evals': evals
    }
    # save_task(task_data)  
    return evals

# if __name__ == "__main__":
#     import uvicorn
#     # 访问127.0.0.1:8000/docs查看文档
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level='debug')