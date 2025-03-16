#!/bin/bash

# 创建必要的目录
mkdir -p logs
mkdir -p res
mkdir -p data
mkdir -p results
mkdir -p models
mkdir -p utils
mkdir -p tests/test_images

# 创建必要的空文件
touch models/__init__.py
touch utils/__init__.py

# 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "环境设置完成！"