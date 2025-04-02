"""
放一些工具函数如保存数据
"""

import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path) 

import requests
from config import SYSTEM_CONFIG
from io import BytesIO
from typing import Union,Literal
from hashlib import sha256
from uuid import uuid4
import logging
from PIL import Image

# MODEL_OPTIONS = Literal['MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet']

def setup_logger(name="Backend/Tools", level=SYSTEM_CONFIG["log_level"]):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，默认使用config.py中的配置
    
    Returns:
        配置好的日志记录器
    """
    
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    logging.basicConfig(
        level=level_map.get(level, logging.INFO),
        format=SYSTEM_CONFIG["log_format"],
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(name)

logger = setup_logger()

def get_img(img_url)->Union[Image.Image, None]:
    # 获取图片文件，返回Image对象
    if img_url.startswith("http"): # 如果是URL
        try:
            response = requests.get(img_url) # 下载图像
            response.raise_for_status() # 检查是否下载成功
            image = Image.open(BytesIO(response.content)).convert("RGB") # 打开图像并转换为RGB模式
            logger.info(f"图像 {img_url} 下载成功，大小为 {image.size}")
        except Exception as e:
            logger.error(f"下载图像 {img_url} 失败: {str(e)}")
            return None
        return image
def text_to_dict(text:Union[str] = "")->dict:
    """
    将txt文件读取到的text转为字典
    0:2 Meta: upload_usr_id, img_sha256, img_upload_time
    4:8 ResNet
    10:14 VGG
    16:20 FasterNet
    因为已知文本结构，就直接用索引填空了

    Args:
        text: 文本内容，默认为""

    Returns:
        dict: text为""时返回空字典
    """
    report_dict = {
        "Meta":{
            "upload_usr_id": "",
            "img_sha256": "",
            "img_upload_time": ""
        },
        "ResNet":{
            "created_time": "",
            "task_id": "",
            "protein": 0.0,
            "oil": 0.0,
        },
        "VGG":{
            "created_time": "",
            "task_id": "",
            "protein": 0.0,
            "oil": 0.0,
        },
        "FasterNet":{
            "created_time": "",
            "task_id": "",
            "protein": 0.0,
            "oil": 0.0,
        }
    } # 将文本转为报告字典
    if text != "": # 如果没有传入文本，则返回空字典
        lines = text.splitlines()
        report_dict["Meta"]["upload_usr_id"] = lines[1]
        report_dict["Meta"]["img_sha256"] = lines[2]
        report_dict["Meta"]["img_upload_time"] = lines[3]
        report_dict["ResNet"]["created_time"] = lines[6]
        report_dict["ResNet"]["task_id"] = lines[7]
        report_dict["ResNet"]["protein"] = float(lines[8].split(":")[1].strip())
        report_dict["ResNet"]["oil"] = float(lines[9].split(":")[1].strip())
        report_dict["VGG"]["created_time"] = lines[12]
        report_dict["VGG"]["task_id"] = lines[13]
        report_dict["VGG"]["protein"] = float(lines[14].split(":")[1].strip())
        report_dict["VGG"]["oil"] = float(lines[15].split(":")[1].strip())
        report_dict["FasterNet"]["created_time"] = lines[18]
        report_dict["FasterNet"]["task_id"] = lines[19]
        report_dict["FasterNet"]["protein"] = float(lines[20].split(":")[1].strip())
        report_dict["FasterNet"]["oil"] = float(lines[21].split(":")[1].strip())
    return report_dict

def dict_to_text(report_dict:dict)->str:
    """
    将字典转为文本
    """
    text = ""
    for key, value in report_dict.items():
        text += f"model: {key}\n" if key!="Meta" else "Meta\n"
        if isinstance(value, dict): # 如果是字典，则递归
            for k, v in value.items():
                text += f"{k}: {v}\n"
            text += "\n" # 每个字典后面加一个空行
    return text

def data_query(level:Literal["check_all", "check_user" ,"all","user"], usr_id = None):
    """
    读取本地所有图片
    """
    if "user" in level and usr_id == None:
        logger.warning(f"查询失败：查询级别为{level}，但是未传入usr_id")
        return
    elif "user" not in level and usr_id != None:
        logger.warning(f"查询失败：查询级别为{level}，但是传入usr_id")
    
    db_path = SYSTEM_CONFIG["save_path"]

    # 读取下面的所有文件夹的名字，即包含基本信息
    all_records = os.listdir(db_path)

    if level == "check": # check模式主要用于返回关键信息方便查重
        return all_records
    
    elif level == "all":
        all_uploads = [
                {
                    "image_path": str,
                   "report": dict
                }
            ] 
        for record in all_records:
            file_path = os.path.join(db_path,record)
            files = os.listdir(file_path)
            for f in files: # 遍历两个文件
                if f.endswith(".txt"): # 由于还不确定图片格式，先处理文本文件
                    with open(os.path.join(file_path,f),"r",encoding="utf-8") as f:
                        text = f.read()
                        report_dict = text_to_dict(text)
                else: # 图片文件
                    img_path = os.path.join(file_path,f)
            all_uploads.append({
                "image_path": img_path,
                "report": report_dict 
            })
            
        return all_uploads
    
    elif level == "check_user":
        user_records = [] # 存储用户上传的记录
        for record in all_records:
            if usr_id in record: # 如果是该用户上传的，则记录下来
                user_records.append(record)
        return user_records
    
    elif level == "user":
        user_uploads = [
                {
                    "image_path": str, 
                    "report": dict
                }
            ]
        for record in all_records:
            if usr_id in record:
                # 如果是该用户上传的，则组合路径读取下面的文件一起返回
                file_path = os.path.join(db_path,record)
                files = os.listdir(file_path)
                for f in files: # 遍历两个文件
                    if f.endswith(".txt"): # 由于还不确定图片格式，先处理文本文件
                        with open(os.path.join(file_path,f),"r",encoding="utf-8") as f:
                            text = f.read()
                            report_dict = text_to_dict(text)
                    else: # 图片文件
                        img_path = os.path.join(file_path,f)
                user_uploads.append({
                    "image_path": img_path,
                    "report": report_dict 
                })
        return user_uploads

def img_hash(image:Image.Image):
    """
    生成图片sha256
    读取文件夹路径看是否重复
    检查图片是否重复
    
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
    
    task_data = {
        "usr_id": str 用户id
        "image" : Image 图片对象
        "model" : 模型
        "evals" : dict
    }
    report = {
        "model_name": VGG|ResNet...
        "timestamp": datetime 采摘时间
        "created_time": datetime 报告生成时间
        "protein": float 蛋白质指标
        "oil": float 油脂指标
    }

    report.txt格式
    上传用户: usrid 接口传参
    图片哈希: hash本地生成
    上传时间: timestamp 接口传参

    model: ResNet
    task_id: uid本地生成
    
    设计结构
    db目录下
    usrid_imgNo 用户id+上传的第几个图片
    |__datetime.jpg （重命名为哈希值）
    |__report.txt
    """
    db_path = SYSTEM_CONFIG["save_path"]
    img_ha = img_hash(task_data["image"]) # 图片哈希值

    user_records = data_query("check_user",usr_id=task_data["usr_id"])
    for index,record in enumerate(user_records):
        if img_ha in record:
            logger.info(f"数据 {record} 重复")
            record_path = os.path.join(db_path,record) # 记录路径
            files = os.listdir(record_path) # 读取文件夹下的文件
            for f in files: # 遍历两个文件
                if f.endswith(".txt"): # 由于还不确定图片格式，先处理文本文件
                    with open(os.path.join(record_path,f),"r",encoding="utf-8") as f:
                        text = f.read()
                        report_dict = text_to_dict(text)
            break
    else: # 如果循环正常结束，则说明没有重复的
        no = len(user_records) + 1 # 图片序号
        report_dict = text_to_dict() # 初始化报告字典
        report_dict["Meta"]["img_upload_time"] = task_data["timestamp"] # 上传时间
        record = f"{task_data['usr_id']}_{no}_{img_ha}" # 记录路径
        record_path = os.path.join(db_path,record) # 记录路径
        os.makedirs(record_path, exist_ok=True) # 创建目录

        image_path = os.path.join(record_path,f"{task_data['timestamp']}.jpg") # 图片路径
        task_data["image"].save(image_path) # 保存图片
        
    report_dict["Meta"]["upload_usr_id"] = task_data["usr_id"] # 上传用户id
    
    report_dict["Meta"]["img_sha256"] = img_ha
    
    if task_data["model"] not in report_dict: # 如果模型不在报告字典中，则报错并在末尾添加
        logger.error(f"模型 {task_data['model']} 不在规定的模型列表中，依然保存数据但是数据无法正常读取，请等手动处理")

    report_dict[task_data["model"]]["created_time"] = task_data["timestamp"] # 报告生成时间
    report_dict[task_data["model"]]["task_id"] = uuid4() # 任务id
    report_dict[task_data["model"]]["protein"] = task_data["evals"]["protein"] # 蛋白质指标
    report_dict[task_data["model"]]["oil"] = task_data["evals"]["oil"] # 油脂指标
    report_path = os.path.join(record_path,f"report.txt") # 报告路径
    with open(report_path,"w",encoding="utf-8") as f: # 保存报告
        f.write(dict_to_text(report_dict))

    logger.info(f"数据 {record} 保存成功")
