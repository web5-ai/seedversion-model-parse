#!/bin/bash

# 定义需要检查的软件列表
required_software=("tar" "python3" "python3-venv" "systemctl")
echo "正在检查系统环境..."
# 检查并安装软件
for software in "${required_software[@]}"; do
    if ! command -v "$software" &> /dev/null; then
        echo "$software 未安装，正在尝试安装..."
        sudo apt-get update
        if [ "$software" = "systemctl" ]; then
            # systemctl 是 systemd 的一部分，通常系统自带，若有问题可提示用户手动处理
            echo "systemctl 是 systemd 的一部分，若安装失败请手动检查系统配置。"
        else
            sudo apt-get install -y "$software"
            if [ $? -ne 0 ]; then
                echo "安装 $software 失败，请手动处理。"
                exit 1
            fi
        fi
    fi
done

echo "系统环境检查完成，开始解压和安装..."
# 定义项目路径
project_name="seedversion-model-parse"
base_path="/home/www"
project_path="$base_path/$project_name"
# 创建项目目录
if [ ! -d "$project_path" ]; then
    echo "创建项目目录：$project_path"
    mkdir -p "$project_path"
else
    echo "项目目录已存在：$project_path"
fi

# 定义压缩文件名称
zip_file="seedversion-model-parse.tar"

# 进入目标路径
if [ ! -d "$base_path" ]; then
    echo "$base_path 目录不存在"
fi
cd $base_path
echo "当前目录为：$PWD"
# 解压文件 到指定目录
# 假设 tar 文件没有经过额外压缩

# tar -xf $zip_file -C $project_path

# 进入项目目录
echo "正在进入项目目录 $project_name"
if [ ! -d "$project_name" ]; then
    echo "项目目录不存在，请检查解压路径。"
    exit 1
fi
cd $project_name
echo "当前目录为：$PWD"
# 创建虚拟环境
python3 -m venv env

# 激活虚拟环境
source env/bin/activate

# 升级 pip
# pip install --upgrade pip

# 安装项目依赖
if [ -f requirements.txt ]; then
    echo "正在安装依赖..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "未找到 requirements.txt 文件，跳过依赖安装步骤。"
    exit 1
fi

# 退出虚拟环境
deactivate

# 获取虚拟环境中的 Python 解释器路径
python_path="$project_path/env/bin/python"

# 创建 systemd 服务文件
sudo bash -c "cat > /etc/systemd/system/$project_name.service << EOF
[Unit]
Description=$project_name based on FastAPI application server
After=network.target

[Service]
User=\$USER
Group=\$USER
WorkingDirectory=$project_path
Environment=\"PATH=$project_path/env/bin:\$PATH\"
ExecStart=$python_path $project_path/backend/run.py

[Install]
WantedBy=multi-user.target
EOF"

# 重新加载 systemd 管理器配置
sudo systemctl daemon-reload

# 启动服务并设置开机自启
sudo systemctl start $project_name
sudo systemctl enable $project_name

# 检查服务状态
service_status=$(sudo systemctl is-active $project_name)
if [ "$service_status" = "active" ]; then
    echo "$project_name 服务已成功启动。"
else
    echo "$project_name 服务启动失败，请检查配置和日志。"
fi