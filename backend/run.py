'''
用于启动后台
'''

import uvicorn
from datetime import datetime
import os
import psutil
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import sys
import time
import json
import torch
from config import SYSTEM_CONFIG, BACKEND_CONFIG
from utils.logging_config import get_logger
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 添加这行解决OpenMP冲突
def init_logging():

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = os.path.join(BACKEND_CONFIG['log_dir'], timestamp)
    # BACKEND_CONFIG["images_dir"] = os.path.join(log_path, "images") # 每次启动时更新图像目录
    # 检查日志路径是否存在，如果不存在则创建
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 只使用一个日志文件
    backend_log = os.path.join(log_path, "backend.log")

    # 配置uvicorn日志
    logging_config = {
        "version": 1,
        # 禁用已有的日志器，防止重复输出
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": BACKEND_CONFIG["log_format"],
                "datefmt": BACKEND_CONFIG["log_date_format"],
            },
        },
        "handlers": {
            "file": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": backend_log,
                "encoding": BACKEND_CONFIG["log_encoding"],
            },
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "Backend": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "ForceEnv": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "ModelAPI": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "Backend/Tools": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    return logging_config, log_path

# 从配置中获取内存监控和邮件配置
MEMORY_LIMIT = BACKEND_CONFIG["memory_limit"]
SMTP_SERVER = BACKEND_CONFIG["smtp_server"]
SMTP_PORT = BACKEND_CONFIG["smtp_port"]
SMTP_USERNAME = BACKEND_CONFIG["smtp_username"]
SMTP_PASSWORD = BACKEND_CONFIG["smtp_password"]
RECIPIENT_EMAIL = BACKEND_CONFIG["recipient_email"]

def send_email(subject, message):
    # 获取日志记录器
    logger = get_logger("Backend")

    msg = MIMEText(message, 'plain', 'utf-8')
    msg['From'] = SMTP_USERNAME
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = Header(subject, 'utf-8')

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def monitor_memory():
    # 获取日志记录器
    logger = get_logger("Backend")
    main_process = psutil.Process(os.getpid())
    last_memory_record = 0

    while True:
        total_memory_usage = 0
        # 遍历所有子进程
        for proc in main_process.children(recursive=True):
            try:
                # 累加子进程的内存使用
                total_memory_usage += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        # 加上主进程的内存使用
        total_memory_usage += main_process.memory_info().rss

        if abs(total_memory_usage - last_memory_record) > 1024 * 1024 * 500:  # 500MB
            last_memory_record = total_memory_usage
            # 使用我们的日志记录器
            logger.info(f"Current total memory usage: {total_memory_usage / (1024 * 1024)} MB, last memory record: {last_memory_record / (1024 * 1024)} MB")

        if total_memory_usage > MEMORY_LIMIT:
            logger.warning(f"Memory usage exceeded limit: {total_memory_usage} bytes, restarting the project.")
            send_email("Memory Limit Exceeded", f"Project is restarting due to memory usage exceeding {MEMORY_LIMIT} bytes.")
            # # 重启项目
            # python = sys.executable
            # os.execv(python, [python] + sys.argv)

        # 使用配置中的内存检查间隔
        time.sleep(BACKEND_CONFIG["memory_check_interval"])

# def heartbeat(interval=60):
#     """
#     心跳函数，定期记录日志以避免系统休眠
#     Args:
#         interval: 心跳间隔，单位为秒
#     """
#     uvicorn_logger = logging.getLogger("uvicorn")
#     heartbeat_count = 0
#     while True:
#         heartbeat_count += 1
#         uvicorn_logger.info(f"Heartbeat #{heartbeat_count} - 服务正常运行中")
#         time.sleep(interval)

def run_server():
    logging_config, log_path = init_logging()
    # 127.0.0.1:8000打开网页
    # 访问127.0.0.1:8000/docs查看文档

    # 设置种子
    os.environ["PYTHONHASHSEED"] = str(SYSTEM_CONFIG["default_seed"])
    # 系统环境变量
    print("系统环境变量:")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    # 使用自定义日志配置启动 uvicorn
    print("Starting uvicorn server...")

    # # 启动内存监控线程
    # memory_thread = threading.Thread(target=monitor_memory)
    # memory_thread.daemon = True
    # memory_thread.start()

    # 获取日志记录器
    logger = get_logger("Backend")

    # 记录配置信息到日志（简化版本）
    logger.info("系统已加载配置")

    # 运行环境测试
    logger.info("开始运行环境测试...")
    try:
        # 确保在项目根目录下运行
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(root_dir)
        logger.info(f"切换工作目录到项目根目录: {root_dir}")

        # 导入ModelAPI，确保正确的导入路径
        sys.path.append(root_dir)
        from backend.model_api import ModelAPI

        # 初始化ModelAPI
        logger.info("初始化ModelAPI...")
        model_api = ModelAPI()
        logger.info("ModelAPI初始化完成")

        # 运行环境测试，使用默认模型和种子
        logger.info("运行环境测试...")
        env_test_result = model_api.run_env_test()

        # 将结果保存到日志目录
        env_test_file = os.path.join(log_path, "env_test_result.json")
        with open(env_test_file, 'w', encoding='utf-8') as f:
            json.dump(env_test_result, f, ensure_ascii=False, indent=4)

        # 记录关键信息到日志
        logger.info(f"环境测试结果已保存至: {env_test_file}")
        logger.info(f"模型: {env_test_result['model'].get('name', '未知')}, 参数数量: {env_test_result['model'].get('total_params', '未知')}")
        if 'prediction' in env_test_result and 'protein' in env_test_result['prediction'] and 'oil' in env_test_result['prediction']:
            prediction = env_test_result['prediction']
            logger.info(f"环境测试完成: 蛋白质={prediction['protein']:.4f}, 油脂={prediction['oil']:.4f}, 哈希={prediction.get('hash', '未知')[:8]}...")
        else:
            logger.info("环境测试完成")

        # 记录CUDA设备信息
        if 'environment' in env_test_result and env_test_result['environment'].get('cuda_available', False):
            env = env_test_result['environment']
            logger.info(f"CUDA可用: {env.get('gpu_name', '未知')}")

        # 释放资源
        del model_api
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理CUDA缓存")
    except Exception as e:
        logger.error(f"环境测试失败: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")

    # # 心跳包间隔（秒）- 已注释
    # HEARTBEAT_INTERVAL = 3600
    # uvicorn_logger.info(f"Starting heartbeat thread with interval: {HEARTBEAT_INTERVAL} seconds")

    # # 启动心跳线程，避免系统休眠 - 已注释
    # heartbeat_thread = threading.Thread(target=lambda: heartbeat(interval=HEARTBEAT_INTERVAL))
    # heartbeat_thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
    # heartbeat_thread.start()

    try:
        # 在主线程中运行uvicorn服务器
        uvicorn.run(
            "main:app",
            host=BACKEND_CONFIG["host"],
            port=BACKEND_CONFIG["port"],
            reload=BACKEND_CONFIG["reload"],
            log_config=logging_config
        )
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        sys.exit(0)
