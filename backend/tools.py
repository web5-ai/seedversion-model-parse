"""
放一些工具函数如保存数据
"""

import os
import sys
import pickle
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import requests
from config import SYSTEM_CONFIG, BACKEND_CONFIG
from io import BytesIO
from typing import Union,Literal
from hashlib import sha256
from uuid import uuid4
from PIL import Image
import datetime
import subprocess
from utils.logging_config import get_logger

# MODEL_OPTIONS = Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet']

# 获取日志记录器
logger = get_logger("Tools")

def get_img(img_url:str, ps=True):
    '''
    将原来的函数解耦一些
    '''

    image = download_img(img_url) # 下载图像
    hash256 = img_hash(image) # 计算哈希值

    save_path = None # 保存路径
    if ps: # 如果需要处理
        image, save_path = img_ps(image) # 处理图像
        logger.info(f"图像 {img_url} 处理成功，大小为 {image.size}")
    return image, save_path, hash256

def download_img(img_url)->Union[Image.Image, None]:
    # 获取图片文件，返回Image对象
    if img_url.startswith("http"): # 如果是URL
        try:
            response = requests.get(img_url) # 下载图像
            response.raise_for_status() # 检查是否下载成功
            image = Image.open(BytesIO(response.content)).convert("RGB") # 打开图像并转换为RGB模式
            logger.info(f"图像 {img_url} 下载成功，大小为 {image.size}")
        except Exception as e:
            logger.error(f"下载图像 {img_url} 失败: {str(e)}")
            raise e
        return image
    else: # 如果是本地文件路径
        try:
            # 检查文件是否存在
            if not os.path.exists(img_url):
                logger.error(f"本地图像文件不存在: {img_url}")
                return None

            # 打开本地图像文件
            image = Image.open(img_url).convert("RGB")
            logger.info(f"本地图像 {img_url} 加载成功，大小为 {image.size}")
            return image
        except Exception as e:
            logger.error(f"加载本地图像 {img_url} 失败: {str(e)}")
            return None

def get_detection_counts(results):
    """
    获取检测结果中各类别的数量统计
    
    Args:
        results: ultralytics.engine.results.Results对象
        
    Returns:
        dict: 各类别及其数量的统计字典
    """
    # 初始化类别计数字典
    counts = {}
    
    # 确保results是Results对象
    if hasattr(results, 'boxes') and results.boxes is not None:
        # 获取类别ID和置信度
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy()  # 将类别ID移至CPU并转换为numpy数组
        
        # 获取类别名称映射
        class_names = results.names if hasattr(results, 'names') else {}
        
        # 统计各类别数量
        for class_id in class_ids:
            # 获取类别名称，如果没有则使用ID
            class_name = class_names.get(int(class_id), f'class_{int(class_id)}')
            # 更新计数
            counts[class_name] = counts.get(class_name, 0) + 1
    
    return counts

def img_ps(image:Image.Image):
    '''
    用ps脚本处理图片，这里处理完会保存到本地，返回Image对象和保存路径
    '''

    # js脚本路径
    js_path = os.path.join('backend', '680x680.jsx')
    # ps路径
    ps_path = r'D:\Tools\PS\Adobe Photoshop 2023\Photoshop.exe'
    # 文件夹路径
    images_dir = BACKEND_CONFIG["images_dir"]
    # 保存文件名（日期时间 2025-04-28 17:26:23）
    save_name_before = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ' before.png'  # 使用连字符替换冒号，避免Windows文件名问题
    save_name_after = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") +'after.png'  # 使用连字符替换冒号，避免Windows文件名问题
    # 保存路径
    save_path_before = os.path.join(images_dir, save_name_before)
    save_path_after = os.path.join(images_dir, save_name_after)
    # 先将对象保存到本地
    try:
        image.save(save_path_before)
        logger.info(f"图像已保存到: {save_path_before}")
    except Exception as e:
        logger.error(f"保存图像失败: {str(e)}")
        raise e

    args = [ps_path, js_path, save_path_before, save_path_after]
    # 执行命令行
    try:
        # 使用subprocess.DEVNULL隐藏输出
        subprocess.run(args, creationflags=subprocess.CREATE_NO_WINDOW)
        logger.info(f"PS处理完成: {save_path_after}")
    except Exception as e:
        logger.error(f"PS处理失败: {str(e)}")

    # 读取处理后的图片
    try:
        processed_image = Image.open(save_path_after)
        logger.info(f"图像已读取: {save_path_after}")

        return processed_image, save_path_after
    except Exception as e:
        logger.error(f"读取图像失败: {str(e)}")
        raise e

def data_query(level:Literal["check_all", "check_user" ,"all","user","image"], **kwargs):
    """
    数据查询
    check_all返回所有文件记录，包括用户 序号 哈希，不包含具体文件，主要用来查重
    check_user同上
    all user 返回从pickle加载的字典和图片哈希
    image 通过图片hash查询图片路径
    **kwargs接收usr_id和image_hash两个关键字
    """

    db_path = SYSTEM_CONFIG["save_path"]

    # 读取下面的所有文件夹的名字，即包含基本信息
    all_records = os.listdir(db_path)

    if level == "check_all": # check模式主要用于返回关键信息方便查重
        return all_records

    elif level == "all":
        all_uploads = []
        for record in all_records:
            file_path = os.path.join(db_path,record)
            files = os.listdir(file_path)
            for f in files: # 遍历两个文件
                if f.endswith(".pickle"): # 由于还不确定图片格式，先处理文本文件
                    with open(os.path.join(file_path,f),"rb") as f:
                        report_dict = pickle.load(f)
            all_uploads.append(report_dict)

        return all_uploads

    elif level == "check_user":
        try:
            user_records = [] # 存储用户上传的记录
            for record in all_records:
                if kwargs["usr_id"] in record: # 如果是该用户上传的，则记录下来
                    user_records.append(record)
        except KeyError:
            logger.error("查询用户上传记录时缺少usr_id参数")
        return user_records

    elif level == "user":
        user_uploads = []
        for record in all_records:
            try:
                if kwargs["usr_id"] in record:
                    # 如果是该用户上传的，则组合路径读取下面的文件一起返回
                    file_path = os.path.join(db_path,record)
                    files = os.listdir(file_path)
                    for f in files: # 遍历两个文件
                        if f.endswith(".pickle"): # 由于还不确定图片格式，先处理文本文件
                            with open(os.path.join(file_path,f),"rb") as f:
                                report_dict = pickle.load(f)
                    image_hash = record.split("_")[-1] # 图片哈希值
                    user_uploads.append(report_dict)
            except KeyError:
                logger.error("查询用户上传记录时缺少usr_id参数")

    elif level == "image": # 返回图片路径
        try:
            for record in all_records:
                if kwargs["image_hash"] in record: # 如果是该图片，就返回图片地址
                    file_path = os.path.join(db_path,record)
                    files = os.listdir(file_path)
                    for f in files: # 遍历两个文件
                        if not f.endswith(".pickle"):
                            # 生成路径
                            img_path = os.path.join(file_path,f)
                    return img_path
        except KeyError:
            logger.error("查询图片时缺少image_hash参数")

    return user_uploads

def img_hash(image:Image.Image):
    """
    生成图片sha256
    读取文件夹路径看是否重复

    Args:
        image: Image对象
    Returns:
        str: 图片sha256, 图片序号
    """
    image_bytes = BytesIO()
    image.save(image_bytes, "JPEG")
    image_bytes = image_bytes.getvalue()
    image_hash = sha256(image_bytes).hexdigest()

    return image_hash

def save_task(task_data:dict)->None:
    """
    保存任务
    放弃文本保存转为使用pickle库直接读写
    task_data = {
        "timestamp": datetime 采样时间
        "usr_id": str 用户id
        "image" : Image 图片对象
        "model" : 模型
        "evals" : dict
    }


    设计结构
    db目录下
    usrid_imgNo 用户id+上传的第几个图片
    |__datetime.jpg （重命名为上传时间）
    |__report.pickle （保存报告）
    """
    db_path = SYSTEM_CONFIG["save_path"]
    img_ha = img_hash(task_data["image"]) # 图片哈希值
    report_dict = {
        "Meta":{}
    }
    user_records = data_query("check_user",usr_id=task_data["usr_id"])
    for index,record in enumerate(user_records):
        if img_ha in record:
            no = record.split("_")[1] # 图片序号
            logger.info(f"数据 {record} 重复, 更新数据")
            record_path = os.path.join(db_path,record) # 记录路径
            files = os.listdir(record_path) # 读取文件夹下的文件
            for f in files: # 遍历两个文件
                if f.endswith(".pickle"): # 由于还不确定图片格式，先处理pickle
                    with open(os.path.join(record_path,f),"rb") as f:
                        report_dict = pickle.load(f)
            break
    else: # 如果循环正常结束，则说明没有重复的
        no = len(user_records) + 1 # 图片序号
        report_dict["Meta"]["img_upload_time"] = task_data["timestamp"] # 上传时间
        record = f"{task_data['usr_id']}_{no}_{img_ha}" # 记录路径
        record_path = os.path.join(db_path,record) # 记录路径
        os.makedirs(record_path, exist_ok=True) # 创建目录

        image_path = os.path.join(record_path,f"{task_data['timestamp']}.jpg") # 图片路径
        task_data["image"].save(image_path) # 保存图片

    report_dict.update({task_data["model"]:{}})

    report_dict["Meta"]["upload_usr_id"] = task_data["usr_id"] # 上传用户id
    report_dict["Meta"]["no"] = no # 图片序号
    report_dict["Meta"]["img_sha256"] = img_ha

    report_dict[task_data["model"]]["created_time"] = task_data["timestamp"] # 报告生成时间
    report_dict[task_data["model"]]["task_id"] = uuid4() # 任务id
    report_dict[task_data["model"]]["protein"] = task_data["evals"]["protein"] # 蛋白质指标
    report_dict[task_data["model"]]["oil"] = task_data["evals"]["oil"] # 油脂指标
    report_path = os.path.join(record_path,f"report.pickle") # 报告路径
    # 保存为pickle
    with open(report_path,"wb") as f:
        pickle.dump(report_dict,f)

    logger.info(f"数据 {record} 保存成功")