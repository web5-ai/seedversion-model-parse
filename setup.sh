#!/bin/bash

# 创建必要的目录
mkdir -p logs
mkdir -p res
mkdir -p data

# 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt