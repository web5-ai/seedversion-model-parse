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

# 生成带时间戳的日志文件名
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# 检查日志路径是否存在，如果不存在则创建
if not os.path.exists("./logs"):
    os.makedirs("./logs")

log_filename = f"./logs/backend_{timestamp}.log"

# 配置日志
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
            "filename": log_filename,
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
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def monitor_memory():
    main_process = psutil.Process(os.getpid())
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
        # 实时打印当前内存消耗
        logging.info(f"Current total memory usage: {total_memory_usage} bytes")
        if total_memory_usage > MEMORY_LIMIT:
            logging.warning(f"Memory usage exceeded limit: {total_memory_usage} bytes, restarting the project.")
            send_email("Memory Limit Exceeded", f"Project is restarting due to memory usage exceeding {MEMORY_LIMIT} bytes.")
            # # 重启项目
            # python = sys.executable
            # os.execv(python, [python] + sys.argv)

if __name__ == "__main__":
    # 127.0.0.1:8000打开网页
    # 访问127.0.0.1:8000/docs查看文档
    # 启动内存监控线程
    memory_monitor = threading.Thread(target=monitor_memory)
    memory_monitor.start()
    # 使用自定义日志配置启动 uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_config=logging_config)