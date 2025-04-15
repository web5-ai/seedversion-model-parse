'''
用于启动后台
'''

import uvicorn
import logging
from datetime import datetime
import os
import psutil
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import threading
import sys
import time

# 生成带时间戳的日志文件名
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
log_path = f"./logs/{timestamp}"
# 检查日志路径是否存在，如果不存在则创建
if not os.path.exists(log_path):
    os.makedirs(log_path)

# 只使用一个日志文件
backend_log = f"{log_path}/backend.log"

# 配置uvicorn日志
logging_config = {
    "version": 1,
    # 禁用已有的日志器，防止重复输出
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "file": {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": backend_log,
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
            # 防止日志向上传播到根日志器
            "propagate": False,
        },
    },
}

# 配置内存监控
MEMORY_LIMIT = 1024 * 1024 * 10000  # 10000MB
# 配置邮件信息
SMTP_SERVER = 'smtp.qq.com'
SMTP_PORT = 587
SMTP_USERNAME = '851680026@qq.com'
SMTP_PASSWORD = 'krrlmqusmxkdbcdg'
RECIPIENT_EMAIL = '851680026@qq.com'

def send_email(subject, message):
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
        logging.getLogger("uvicorn").info("Email sent successfully.")
    except Exception as e:
        logging.getLogger("uvicorn").error(f"Failed to send email: {e}")

def monitor_memory():
    # 获取uvicorn的日志记录器
    uvicorn_logger = logging.getLogger("uvicorn")
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
            # 使用uvicorn的日志记录器
            uvicorn_logger.info(f"Current total memory usage: {total_memory_usage / (1024 * 1024)} MB, last memory record: {last_memory_record / (1024 * 1024)} MB")

        if total_memory_usage > MEMORY_LIMIT:
            uvicorn_logger.warning(f"Memory usage exceeded limit: {total_memory_usage} bytes, restarting the project.")
            send_email("Memory Limit Exceeded", f"Project is restarting due to memory usage exceeding {MEMORY_LIMIT} bytes.")
            # # 重启项目
            # python = sys.executable
            # os.execv(python, [python] + sys.argv)

        # 每隔五秒检查一次
        time.sleep(5)

def heartbeat():
    """
    心跳函数，定期记录日志以避免系统休眠
    """
    # 获取uvicorn的日志记录器
    uvicorn_logger = logging.getLogger("uvicorn")
    heartbeat_count = 0
    while True:
        heartbeat_count += 1
        uvicorn_logger.info(f"Heartbeat #{heartbeat_count} - 服务正常运行中")
        # 每60秒发送一次心跳
        time.sleep(60)

def run_server():
    # 127.0.0.1:8000打开网页
    # 访问127.0.0.1:8000/docs查看文档

    # 使用自定义日志配置启动 uvicorn
    print("Starting uvicorn server...")

    # 启动uvicorn服务器
    server_thread = threading.Thread(
        target=lambda: uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_config=logging_config)
    )
    server_thread.daemon = True
    server_thread.start()

    # 等待uvicorn启动完成
    time.sleep(2)

    # 获取uvicorn的日志记录器
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.info("Starting heartbeat thread")

    # 启动心跳线程，避免系统休眠
    heartbeat_thread = threading.Thread(target=heartbeat)
    heartbeat_thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
    heartbeat_thread.start()

    # 防止主线程退出
    try:
        while True:
            time.sleep(60)  # 每小时检查一次
    except KeyboardInterrupt:
        uvicorn_logger.info("Server shutting down...")
        sys.exit(0)
