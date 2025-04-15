import asyncio
import logging
from fastapi import FastAPI
from fastapi.responses import FileResponse
from type_cls import TaskModel
from tools import *
from model_api import ModelAPI
import datetime
from run import run_server

# 获取uvicorn的日志记录器
logger = logging.getLogger("uvicorn")

app = FastAPI()

# 初始化模型API，加载默认模型
model_api = ModelAPI()

@app.get("/")
def root():
    logger.info("访问根路径")
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
        image_src: 图像文件的URL或路径
        model_name: Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet']

    Returns:
        {
            "oil": float,
            "protein": float,
            'time_delta': float, # 模型预测时间，单位为s
            'memory_cost': float, # 模型预测内存消耗，单位为MB
        }
    '''
    logger.info(f"收到预测请求: 模型={task_info.model_name}, 图像源={task_info.img_src}")
    
    # 读取上传的图像文件，使用asyncio.to_thread在单独线程中运行同步函数
    try:
        logger.info("开始下载/读取图像")
        img = await asyncio.to_thread(get_img, task_info.img_src)
        logger.info("图像获取成功")
    except Exception as e:
        logger.error(f"图像获取失败: {str(e)}")
        return {"模型预测失败_图片下载失败": str(e)}
    
    # 模型预测，使用asyncio.to_thread在单独线程中运行同步函数
    try:
        logger.info(f"开始使用{task_info.model_name}模型进行预测")
        s = datetime.datetime.now()
        evals = await asyncio.to_thread(model_api.eval_image, img, task_info.model_name)  # 得到字典
        e = datetime.datetime.now()
        time_delta = (e - s).total_seconds()
        evals['time_delta'] = time_delta
        evals['model'] = task_info.model_name
        logger.info(f"{task_info.model_name}模型预测完成，耗时: {time_delta}秒，结果如下")
        for k,v in evals.items():
            logger.info(f"{k}: {v}")
    except Exception as e:
        logger.error(f"模型预测失败: {str(e)}")
        return {"模型预测失败": str(e)}
    
    # 保存任务到数据库的代码已被注释，如果需要可以取消注释并添加日志
    
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

if __name__ == "__main__":
    run_server()
