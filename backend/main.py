import asyncio
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
    return {"油菜籽fastapi后台"}

# @app.get("/image")
# def get_image(img_hash:str):
#     '''
#     获取图片
#     '''
#     img_path = data_query("image",image_hash=img_hash)

#     return FileResponse(img_path)

@app.post("/predict")
async def predict(task_info: TaskModel) -> dict:
    '''
    预测接口，接收图像文件和模型名称，返回预测结果。
    Args:
        timestamp: 任务上传时间（采样时间）需要约定时间格式便于转换存储
        usr_id: 用户id
        image_src: 图像文件的URL或路径
        model_name: Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet']

    Returns:
        {
            "oil": float,
            "protein": float,
        }
    '''
    # 还没有约定好时间传递方式，不好转换，这里我先自定一个iso格式转换
    # timestamp = task_info.timestamp.strftime("%Y年%m月%d日%H时%M分%S秒")
    
    # 读取上传的图像文件，使用asyncio.to_thread在单独线程中运行同步函数
    try:
        img = await asyncio.to_thread(get_img, task_info.img_src)
    except Exception as e:
        return {"模型预测失败_图片下载失败": e}
    # 模型预测，使用asyncio.to_thread在单独线程中运行同步函数
    try:
        s = datetime.datetime.now()
        evals = await asyncio.to_thread(model_api.eval_image, img, task_info.model_name)  # 得到字典
        e = datetime.datetime.now()
        evals['time_delta'] = (e - s).total_seconds()
    except Exception as e:
        return {"模型预测失败": e}
    # 保存任务到数据库，返回任务id，用于查询任务状态和结果
    # task_data = {
    #     'timestamp': timestamp,
    #     'usr_id': task_info.usr_id,
    #     'image': img,
    #     'model': task_info.model_name,
    #     'evals': evals
    # }
    # 使用asyncio.to_thread在单独线程中运行同步函数
    # await asyncio.to_thread(save_task, task_data)  # 保存为pickle
    
    return evals

# @app.get("/history")
# def history(usr_id:str)->list:
#     '''
#     查询历史任务，返回任务id和任务状态。
#     Args:
#         usr_id: 用户id，str类型，为all时返回全部历史记录
#     Returns:
#         返回用户历史上传
#         [
#             {
#                 "Meta": {
#                 "img_upload_time": "2025年04月07日05时48分13秒",
#                 "upload_usr_id": "111",
#                 "img_sha256": "f8fa11de2237940735980c439fd6f373b40508f2cdc3d52522121b70b18e075a",
#                 "no": "1"
#                 },
#                 "ResNet": {
#                 "created_time": "2025年04月07日05时48分13秒",
#                 "task_id": "ff3b7df5-5f13-43e3-abd9-b28ef0dca42a",
#                 "protein": 49.78548812866211,
#                 "oil": 49.52592086791992
#                 },
#                 "Swin": {
#                 "created_time": "2025年04月07日05时56分09秒",
#                 "task_id": "95b3f67d-4162-46b2-911e-872285a9df21",
#                 "protein": 38.18247985839844,
#                 "oil": 57.29891586303711
#                 }
#             },
#             {
#                 "Meta": {
#                 "img_upload_time": "2025年04月07日05时48分13秒",
#                 "upload_usr_id": "111",
#                 "img_sha256": "f8fa11de2237940735980c439fd6f373b40508f2cdc3d52522121b70b18e075a",
#                 "no": "1"
#                 },
#                 "ResNet": {
#                 "created_time": "2025年04月07日05时48分13秒",
#                 "task_id": "ff3b7df5-5f13-43e3-abd9-b28ef0dca42a",
#                 "protein": 49.78548812866211,
#                 "oil": 49.52592086791992
#                 },
#                 "Swin": {
#                 "created_time": "2025年04月07日05时56分09秒",
#                 "task_id": "95b3f67d-4162-46b2-911e-872285a9df21",
#                 "protein": 38.18247985839844,
#                 "oil": 57.29891586303711
#                 }
#             },
#         ]
#     '''
#     if usr_id == "all":
#         user_uploads = data_query("all") # 得到所有用户上传的所有图片和值
#     # 读取数据库，返回任务id和任务状态
#     else:
#         user_uploads = data_query("user",usr_id=usr_id) # 得到用户上传的所有图片和值
#     return user_uploads

# if __name__ == "__main__":
#     import uvicorn
#     # 访问127.0.0.1:8000/docs查看文档
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level='debug')