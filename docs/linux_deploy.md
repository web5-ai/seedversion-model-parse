# Linux 部署说明

本文档用于在一台全新的 Linux 服务器上部署当前项目，并启动 `FastAPI` 服务。

## 1. 机器准备

建议环境：

- Ubuntu 20.04 / 22.04 / 24.04
- Python 3.10+
- 至少 8GB 内存
- 至少 10GB 可用磁盘
- 如需 GPU，安装好 NVIDIA 驱动和对应 CUDA

## 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y git curl wget build-essential python3 python3-pip python3-venv
```

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

如果 `source ~/.local/bin/env` 不生效，可以重新登录终端后再执行。

## 3. 拉代码

```bash
git clone <你的仓库地址> seedversion-model-parse
cd seedversion-model-parse
```

如果你是把当前目录直接打包上传到服务器，也可以跳过这一步。

## 4. 准备模型文件

当前项目运行至少需要把模型权重放到 `weights/` 目录下。

本次新增的菜籽判别模型必须有：

```bash
mkdir -p weights
cp /你的来源路径/padim_oilseed_model.pth weights/padim_oilseed_model.pth
```

如果还要使用原有接口，你还需要把原项目已有模型一并放进去，例如：

- `weights/3cls.onnx`
- `weights/fruit_ripeness_model.pth`
- 其他成分分析模型文件

是否缺文件，可以看 [config.py](/Users/wanglu/Work/remote/seedversion-model-parse/config.py) 里的配置项。

## 5. 安装 Python 依赖

在项目根目录执行：

```bash
uv sync
```

只启动服务时，默认用：

```bash
uv run python backend/main.py
```

## 6. 启动服务

项目默认监听 `8123` 端口。

前台启动：

```bash
uv run python backend/main.py
```

启动后本机检查：

```bash
curl http://127.0.0.1:8123/
```

菜籽判别接口检查：

```bash
curl -X POST "http://127.0.0.1:8123/rapeseed/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/demo.jpg"
  }'
```

## 7. 用 systemd 托管

创建服务文件：

```bash
sudo tee /etc/systemd/system/seedversion-model-parse.service >/dev/null <<'EOF'
[Unit]
Description=seedversion-model-parse
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/seedversion-model-parse
ExecStart=/home/ubuntu/.local/bin/uv run python backend/main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
```

注意替换下面两个值：

- `User=ubuntu`
- `WorkingDirectory=/home/ubuntu/seedversion-model-parse`

启动并设置开机自启：

```bash
sudo systemctl daemon-reload
sudo systemctl enable seedversion-model-parse
sudo systemctl start seedversion-model-parse
sudo systemctl status seedversion-model-parse
```

查看日志：

```bash
journalctl -u seedversion-model-parse -f
```

## 8. 对外开放端口

如果服务器开启了防火墙，需要放行 `8123`：

```bash
sudo ufw allow 8123/tcp
sudo ufw status
```

云服务器还需要同时在安全组里放行 `8123`。

## 9. 常见问题

### 1) `uv sync` 失败

先看 Python 版本：

```bash
python3 --version
uv python list
```

如果系统 Python 太旧，直接让 `uv` 管理解释器即可。

### 2) 启动时报模型文件不存在

检查：

```bash
ls -lh weights
```

重点确认：

- `weights/padim_oilseed_model.pth`
- 其他被 `config.py` 引用的权重文件

### 3) 首次推理很慢

首次启动或首次调用 `PaDiM` 时，`torchvision` 可能会下载 `resnet34` 预训练权重。这要求服务器第一次联网成功一次。

### 4) GPU 没生效

先检查：

```bash
nvidia-smi
```

如果驱动和 CUDA 没配好，当前项目会回退到 CPU。

## 10. 推荐部署流程

新 Linux 机器上按下面顺序执行最稳：

```bash
sudo apt update
sudo apt install -y git curl wget build-essential python3 python3-pip python3-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
git clone <你的仓库地址> seedversion-model-parse
cd seedversion-model-parse
mkdir -p weights
cp /你的模型目录/* weights/
uv sync
uv run python backend/main.py
```
