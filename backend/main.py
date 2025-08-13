# 强制设置环境，确保在不同启动方式下获得一致的结果
# 必须在所有其他导入之前执行
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 添加这行解决OpenMP冲突
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 首先设置正确的工作目录
import os
import sys

# 获取项目根目录的绝对路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python路径
sys.path.insert(0, ROOT_DIR)
# 切换到项目根目录
os.chdir(ROOT_DIR)

# 导入集中的日志配置
from utils.logging_config import get_logger

# 获取日志记录器
logger = get_logger("Backend")

# 使用统一的环境管理
from utils.environment import initialize_environment

# 初始化环境（包含所有必要的检查和设置）
env_result = initialize_environment(seed=123, verbose=False)

if not env_result.get("success", False):
    logger.error("环境初始化失败，服务无法启动")
    sys.exit(1)

# 其他导入
import asyncio
from fastapi import FastAPI
from type_cls import TaskModel
from tools import *
from model_api import ModelAPI
import datetime
from config import SYSTEM_CONFIG
from run import run_server

app = FastAPI()

# 初始化模型API，加载默认模型
logger.info("初始化ModelAPI...")
model_api = ModelAPI('cuda')
logger.info("ModelAPI初始化完成")

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
    # 记录系统环境中的随机因素，用来排查启动方式差异
    model_api.set_seed(SYSTEM_CONFIG['default_seed'])
    seed_info = model_api.get_seed_info()
    # 用json格式标准输出
    logger.info(f"系统环境中的随机因素: {seed_info}")
    # 读取上传的图像文件，使用asyncio.to_thread在单独线程中运行同步函数
    try:
        logger.info("开始下载/读取图像")
        img, save_path, hash256 = await asyncio.to_thread(get_img, task_info.img_src, False) # 路径先留着，不一定用得到
        logger.info("图像获取成功，hash256: " + hash256)
    except Exception as e:
        logger.error(f"图像获取失败: {str(e)}")
        return {"模型预测失败_图片下载失败": str(e)}

    # 模型预测，使用asyncio.to_thread在单独线程中运行同步函数
    try:
        logger.info(f"开始使用{task_info.model_name}模型进行预测")

        s = datetime.datetime.now()
        evals = model_api.eval_image(img, task_info.model_name)  # 得到字典
        # evals = await asyncio.to_thread(model_api.eval_image, img, task_info.model_name)  # 得到字典
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
