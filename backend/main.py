# 强制设置环境，确保在不同启动方式下获得一致的结果
# 必须在所有其他导入之前执行
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 添加这行解决OpenMP冲突
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 获取项目根目录的绝对路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python路径
sys.path.insert(0, ROOT_DIR)
# 切换到项目根目录
os.chdir(ROOT_DIR)

# 导入集中的日志配置
from utils.logging_config import get_logger

# 获取日志记录器 - Backend模块统一使用Backend标识
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
from type_cls import TaskModel, DetectAndEvalModel, DetectAndEvalModel3, RipenessModel
from tools import *
from model_api import ModelAPI
import datetime
from config import SYSTEM_CONFIG
from run import run_server

app = FastAPI()

# 初始化模型API，加载默认模型（日志在ModelAPI内部处理）
model_api = ModelAPI('cuda')

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
    logger.info(f"收到预测请求: 模型={task_info.model}, 图像源={task_info.image_url}")
    # 记录系统环境中的随机因素，用来排查启动方式差异
    model_api.set_seed(SYSTEM_CONFIG['default_seed'])
    seed_info = model_api.get_seed_info()
    # 用json格式标准输出
    logger.info(f"系统环境中的随机因素: {seed_info}")
    # 读取上传的图像文件，使用asyncio.to_thread在单独线程中运行同步函数
    try:
        logger.info("开始下载/读取图像")
        img, save_path, hash256 = await asyncio.to_thread(get_img, task_info.image_url, False) # 路径先留着，不一定用得到
        logger.info("图像获取成功，hash256: " + hash256)
    except Exception as e:
        logger.error(f"图像获取失败: {str(e)}")
        return {"模型预测失败_图片下载失败": str(e)}

    # 模型预测，使用asyncio.to_thread在单独线程中运行同步函数
    try:
        logger.info(f"开始使用{task_info.model}模型进行预测")

        s = datetime.datetime.now()
        evals = model_api.eval_image(img, task_info.model)  # 得到字典
        # evals = await asyncio.to_thread(model_api.eval_image, img, task_info.model)  # 得到字典
        e = datetime.datetime.now()
        time_delta = (e - s).total_seconds()
        evals['time_delta'] = time_delta
        evals['model'] = task_info.model
        logger.info(f"{task_info.model}模型预测完成，耗时: {time_delta}秒，结果如下")
        for k,v in evals.items():
            logger.info(f"{k}: {v}")
    except Exception as e:
        logger.error(f"模型预测失败: {str(e)}")
        return {"模型预测失败": str(e)}

    # 保存任务到数据库的代码已被注释，如果需要可以取消注释并添加日志

    return evals

@app.post("/v1/predict")
async def predict_v1(task_info: DetectAndEvalModel) -> dict:
    '''
    V1预测接口：智能检测和分析，返回简化结果

    Args:
        img_src: 图像文件的URL或路径
        model_name: 用于成分分析的模型名称，默认为'FasterNet'
        conf_threshold: 检测置信度阈值，None表示使用配置默认值
        iou_threshold: IoU阈值，None表示使用配置默认值

    Returns:
        {
            "object_classes_counts": {},          # 检测到的种子类别统计
            "protein": float,          # 蛋白质含量（如果检测到）
            "oil": float,              # 油脂含量（如果检测到）
            "message": str,            # 状态消息
            "time_delta": float        # 总耗时（秒）
        }
    '''
    logger.info(f"收到检测和评估请求: 模型={task_info.model}, 图像源={task_info.image_url}")
    logger.info(f"检测参数: conf_threshold={task_info.conf_threshold}, iou_threshold={task_info.iou_threshold}")

    # 设置随机种子
    model_api.set_seed(SYSTEM_CONFIG['default_seed'])
    seed_info = model_api.get_seed_info()
    logger.info(f"系统环境中的随机因素: {seed_info}")

    # 读取图像文件
    try:
        logger.info("开始下载/读取图像")
        img, save_path, hash256 = await asyncio.to_thread(get_img, task_info.image_url, False)
        logger.info("图像获取成功，hash256: " + hash256)
    except Exception as e:
        logger.error(f"图像获取失败: {str(e)}")
        return {
            "object_classes_counts": {},
            "protein": 0.0,
            "oil": 0.0,
            "message": f"图像获取失败: {str(e)}",
            "time_delta": 0.0
        }

    # 执行检测和评估
    try:
        logger.info("开始执行检测和评估流程")
        result = model_api.detect_and_eval(
            img,
            task_info.model,
            task_info.conf_threshold,
            task_info.iou_threshold
        )

        # 转换为v1格式的简化结果
        v1_result = {
            "object_classes_counts": result.get("object_classes_counts", {}),
            "protein": 0.0,
            "oil": 0.0,
            "message": result.get("message", ""),
            "time_delta": result.get("total_time_delta", 0.0)
        }

        # 如果检测到对象且有评估结果，提取数值
        if result.get("object_classes_counts") != {}:
            v1_result["protein"] = result.get("protein", 0.0)
            v1_result["oil"] = result.get("oil", 0.0)

        # 记录结果
        if result.get("success"):
            if result.get("object_classes_counts") != {}:
                logger.info(f"检测和评估完成: 检测到对象，蛋白质: {v1_result['protein']:.2f}%, 油脂: {v1_result['oil']:.2f}%, 耗时: {v1_result['time_delta']:.3f}秒")
            else:
                logger.info(f"检测完成但未发现种子对象，耗时: {v1_result['time_delta']:.3f}秒")
        else:
            logger.error(f"检测和评估失败: {result.get('error', '未知错误')}")

        return v1_result

    except Exception as e:
        logger.error(f"检测和评估过程失败: {str(e)}")
        return {
            "object_classes_counts": {},
            "protein": 0.0,
            "oil": 0.0,
            "message": f"处理失败: {str(e)}",
            "time_delta": 0.0
        }

@app.post("/v2/predict")
async def predict_v2(task_info: DetectAndEvalModel) -> dict:
    '''
    V2预测接口：智能检测和分析，返回完整结果详情

    Args:
        img_src: 图像文件的URL或路径
        model_name: 用于成分分析的模型名称，默认为'FasterNet'
        conf_threshold: 检测置信度阈值，None表示使用配置默认值
        iou_threshold: IoU阈值，None表示使用配置默认值

    Returns:
        {
            "success": bool,           # 整体操作是否成功
            "object_classes_counts": {},          # 检测到的种子类别统计
            "message": str,            # 状态消息
            "objects": list,           # 检测到的对象列表
            "protein": float,          # 蛋白质含量
            "oil": float,              # 油脂含量
            "time_delta": float        # 总耗时（秒）
        }
    '''
    logger.info(f"收到V2预测请求: 模型={task_info.model}, 图像源={task_info.image_url}")
    logger.info(f"检测参数: conf_threshold={task_info.conf_threshold}, iou_threshold={task_info.iou_threshold}")

    # 设置随机种子
    model_api.set_seed(SYSTEM_CONFIG['default_seed'])
    seed_info = model_api.get_seed_info()
    logger.info(f"系统环境中的随机因素: {seed_info}")

    # 读取图像文件
    try:
        logger.info("开始下载/读取图像")
        img, save_path, hash256 = await asyncio.to_thread(get_img, task_info.image_url, False)
        logger.info("图像获取成功，hash256: " + hash256)
    except Exception as e:
        logger.error(f"图像获取失败: {str(e)}")
        return {
            "success": False,
            "object_classes_counts": False,
            "message": f"图像获取失败: {str(e)}",
            "objects": [],
            "protein": 0.0,
            "oil": 0.0,
            "time_delta": 0.0
        }

    # 执行检测和评估
    try:
        logger.info("开始执行检测和评估流程")
        result = model_api.detect_and_eval(
            img,
            task_info.model,
            task_info.conf_threshold,
            task_info.iou_threshold
        )

        # 记录结果
        if result.get("success"):
            if result.get("object_classes_counts") != {}:
                logger.info(f"V2检测和评估完成: 检测到 {len(result.get('objects', []))} 个对象，蛋白质: {result.get('protein', 0):.2f}%, 油脂: {result.get('oil', 0):.2f}%, 耗时: {result.get('time_delta', 0):.3f}秒")
            else:
                logger.info(f"V2检测完成但未发现种子对象，耗时: {result.get('time_delta', 0):.3f}秒")
        else:
            logger.error(f"V2检测和评估失败: {result.get('message', '未知错误')}")

        return result

    except Exception as e:
        logger.error(f"V2检测和评估过程失败: {str(e)}")
        return {
            "success": False,
            "object_classes_counts": {},
            "message": f"处理失败: {str(e)}",
            "objects": [],
            "protein": 0.0,
            "oil": 0.0,
            "time_delta": 0.0
        }

@app.post("/ripeness/predict")
async def predict_ripeness(task_info: RipenessModel) -> dict:
    '''
    成熟度分类接口：预测油菜籽的成熟度（绿熟、黄熟、完熟）
    先判断是否为油菜籽，再预测成熟度

    Args:
        image_url: 图像文件的URL或路径

    Returns:
        {
            "success": bool,           # 操作是否成功
            "is_rapeseed": bool,        # 是否为油菜籽
            "ripeness_class": str,       # 预测的成熟度类别（绿熟、黄熟、完熟）
            "confidence": float,        # 预测置信度
            "probabilities": dict,       # 各类别的概率
            "message": str,            # 状态消息
            "time_delta": float        # 总耗时（秒）
        }
    '''
    logger.info(f"收到成熟度分类请求: 图像源={task_info.image_url}")

    # 设置随机种子
    model_api.set_seed(SYSTEM_CONFIG['default_seed'])
    seed_info = model_api.get_seed_info()
    logger.info(f"系统环境中的随机因素: {seed_info}")

    # 读取图像文件
    try:
        logger.info("开始下载/读取图像")
        img, save_path, hash256 = await asyncio.to_thread(get_img, task_info.image_url, False)
        logger.info("图像获取成功，hash256: " + hash256)
    except Exception as e:
        logger.error(f"图像获取失败: {str(e)}")
        return {
            "success": False,
            "is_rapeseed": False,
            "ripeness_class": "",
            "confidence": 0.0,
            "probabilities": {},
            "message": f"图像获取失败: {str(e)}",
            "time_delta": 0.0
        }

    # 执行成熟度分类
    try:
        logger.info("开始执行成熟度分类流程")
        result = model_api.predict_ripeness(img)

        # 记录结果
        if result.get("success"):
            if result.get("is_rapeseed"):
                logger.info(f"成熟度分类完成: 类别={result.get('ripeness_class', '')}, 置信度={result.get('confidence', 0):.4f}, 耗时={result.get('time_delta', 0):.3f}秒")
            else:
                logger.info(f"不是油菜籽: {result.get('message', '')}")
        else:
            logger.error(f"成熟度分类失败: {result.get('message', '未知错误')}")

        return result

    except Exception as e:
        logger.error(f"成熟度分类过程失败: {str(e)}")
        return {
            "success": False,
            "is_rapeseed": False,
            "ripeness_class": "",
            "confidence": 0.0,
            "probabilities": {},
            "message": f"处理失败: {str(e)}",
            "time_delta": 0.0
        }

@app.post("/v3/predict")
async def predict_v3(task_info: DetectAndEvalModel3) -> dict:
    '''
    V3预测接口 - 参考demo3实现，使用KNN判断是否为油菜籽并进行成熟度分类

    Args:
        image_url: 图像文件的URL或路径
        conf_threshold: 检测置信度阈值，None表示使用配置默认值
        iou_threshold: IoU阈值，None表示使用配置默认值

    Returns:
        {
            "success": bool,           # 整体操作是否成功
            "is_rapeseed": bool,       # 是否为油菜籽（KNN判断）
            "ripeness_class": str,     # 成熟度类别（绿熟、黄熟、完熟）
            "confidence": float,       # 预测置信度
            "similarity": float,       # KNN平均相似度
            "probabilities": dict,     # 各类别的概率
            "message": str,            # 状态消息
            "time_delta": float        # 总耗时（秒）
        }
    '''
    logger.info(f"收到V3预测请求: 图像源={task_info.image_url}")

    # 设置随机种子
    model_api.set_seed(SYSTEM_CONFIG['default_seed'])
    seed_info = model_api.get_seed_info()
    logger.info(f"系统环境中的随机因素: {seed_info}")

    # 读取图像文件
    try:
        logger.info("开始下载/读取图像")
        img, save_path, hash256 = await asyncio.to_thread(get_img, task_info.image_url, False)
        logger.info("图像获取成功，hash256: " + hash256)
    except Exception as e:
        logger.error(f"图像获取失败: {str(e)}")
        return {
            "success": False,
            "is_rapeseed": False,
            "ripeness_class": "",
            "confidence": 0.0,
            "similarity": 0.0,
            "probabilities": {},
            "message": f"图像获取失败: {str(e)}",
            "time_delta": 0.0
        }

    # 执行成熟度分类（参考demo3）
    try:
        logger.info("开始执行V3成熟度分类流程（参考demo3）")
        result = model_api.predict_ripeness_v3(img)

        # 记录结果
        if result.get("success"):
            if result.get("is_rapeseed"):
                logger.info(f"V3成熟度分类完成: 类别={result.get('ripeness_class', '')}, "
                          f"置信度={result.get('confidence', 0):.4f}, "
                          f"相似度={result.get('similarity', 0):.3f}, "
                          f"耗时={result.get('time_delta', 0):.3f}秒")
            else:
                logger.info(f"V3判断: 输入不是油菜籽，相似度={result.get('similarity', 0):.3f}, "
                          f"耗时={result.get('time_delta', 0):.3f}秒")
        else:
            logger.error(f"V3成熟度分类失败: {result.get('message', '未知错误')}")

        return result

    except Exception as e:
        logger.error(f"V3成熟度分类过程失败: {str(e)}")
        return {
            "success": False,
            "is_rapeseed": False,
            "ripeness_class": "",
            "confidence": 0.0,
            "similarity": 0.0,
            "probabilities": {},
            "message": f"处理失败: {str(e)}",
            "time_delta": 0.0
        }


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
