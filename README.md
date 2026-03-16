# 油菜籽成分分析系统

基于深度学习的油菜籽成分分析系统，通过图像识别技术快速分析油菜籽中的蛋白质和油脂含量。

## 项目背景

本项目旨在通过深度学习技术快速分析油菜籽中的主要成分含量。该系统可以帮助农业工作者和食品加工企业快速评估油菜籽的品质，提高生产效率和产品质量。

## 运行环境

### 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (可选，但推荐使用)
- 依赖库详见requirements.txt

### 硬件要求

- CPU: 支持现代指令集的多核处理器
- 内存: 至少4GB RAM
- GPU: 推荐使用NVIDIA GPU以加速模型推理
- 存储: 至少1GB可用空间

## 快速开始

### 安装依赖

```bash
uv sync
```

### 测试模型

项目提供了多种测试方式，可以通过命令行参数指定不同的功能：

#### 查看可用模型

```bash
python test_model.py list
```

#### 单模型单图像测试

```bash
python test_model.py single --model FasterNet --image tests/test_images/image_custom.png
```

#### 单模型批量图像测试

```bash
python test_model.py batch --model FasterNet --dir tests/images
```

#### 多模型测试

```bash
python test_model.py multi --image tests/test_images/image_custom.png
```

#### 测试种子分类器

```bash
# 测试所有分类器模型
python test_classifiers.py

# 测试单个分类器
python -c "from test_classifiers import test_classifier_model; test_classifier_model('EfficientNetB0Classifier', 'tests/images/100.jpg')"
```

### 启动后端服务

```bash
uv run python backend/main.py
```

或使用批处理脚本：

```bash
backend\start_service.bat
```

### 新增接口

- `POST /rapeseed/predict`: 使用 `PaDiM` 判断输入图像是否为菜籽
- 详细文档见 [docs/API_Documentation.md](/Users/wanglu/Work/remote/seedversion-model-parse/docs/API_Documentation.md)

### Linux 部署

- 新机器部署说明见 [docs/linux_deploy.md](/Users/wanglu/Work/remote/seedversion-model-parse/docs/linux_deploy.md)

## 项目结构

```
项目根目录/
├── config.py                 # 配置文件
├── backend/                  # 后端接口
│   ├── main.py               # FastAPI运行入口
│   ├── run.py                # 项目运行配置
│   ├── model_api.py          # 模型API封装
│   ├── tools.py              # 后台工具函数
│   ├── type_cls.py           # 接口参数类型定义
│   └── start_service.bat     # 项目一键运行脚本
├── models/                   # 模型定义
│   ├── __init__.py           # 模型接口定义
│   ├── model_zoo.py          # 模型定义文件
│   ├── build_mpvit.py        # MPViT模型
│   ├── build_vanillanet.py   # VanillaNet模型
│   ├── build_swinv2.py       # Swin模型
│   └── swinv2_config.py      # Swin模型配置
├── utils/                    # 工具函数
│   ├── environment.py        # 环境检查
│   ├── image_processor.py    # 图像处理
│   ├── model_loader.py       # 模型加载器
│   └── force_env.py          # 环境强制设置
├── tests/                    # 测试文件
│   └── test_images/          # 测试图像
├── weights/                  # 预训练模型权重
├── results/                  # 结果输出目录
├── requirements.txt          # 依赖库列表
└── test_model.py             # 测试脚本
```

## 模型测试工具

`test_model.py` 提供了多种测试功能，支持命令行参数：

### 命令行参数

```
usage: test_model.py [-h] [--seed SEED] [--device {cuda,cpu}]
                    {single,batch,multi,list} ...

油菜籽成分预测模型测试工具

positional arguments:
  {single,batch,multi,list}
                        命令
    single              单模型单图像测试
    batch               单模型批量图像测试
    multi               多模型测试
    list                检查可用模型

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           随机种子
  --device {cuda,cpu}   运行设备
```

### 子命令参数

#### single - 单模型单图像测试

```
usage: test_model.py single [-h] [--model MODEL] [--image IMAGE]

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    模型路径或名称 (默认: FasterNet)
  --image IMAGE    图像路径 (默认: tests/test_images/image_custom.png)
```

#### batch - 单模型批量图像测试

```
usage: test_model.py batch [-h] [--model MODEL] [--dir DIR] [--output OUTPUT]

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    模型路径或名称 (默认: FasterNet)
  --dir DIR        图像目录 (默认: tests/images)
  --output OUTPUT  输出CSV文件路径
```

#### multi - 多模型测试

```
usage: test_model.py multi [-h] [--image IMAGE] [--except [EXCEPT ...]]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         图像路径 (默认: tests/test_images/image_custom.png)
  --except [EXCEPT ...] 排除的模型名称
```

## 关键技术

1. **深度学习模型**: 使用多种预训练模型进行油菜籽的油脂和蛋白质含量预测
2. **模型加载机制**: 实现了灵活的模型加载器，可以自动识别不同格式的模型文件
3. **图像预处理**: 标准化的图像预处理流程，确保输入模型的图像具有一致的格式
4. **环境强制设置**: 确保在不同启动方式下获得一致的预测结果

## 支持的模型

### 🧪 成分分析模型（回归）
- **FasterNet** (默认) - 快速高效的成分预测
- **ResNet** - 经典残差网络架构
- **Swin** - Swin Transformer架构
- **VanillaNet** - 轻量级网络
- **MPViT** - 多尺度视觉Transformer
- **EfficientNet** - 高效网络架构

### 🌱 种子分类模型（分类）
- **EfficientNetB0Classifier** - 推荐使用，99%+准确率，15.3MB
- **ResNet18Classifier** - 快速推理，99%+准确率，42.6MB
- **CustomCNNClassifier** - 轻量级自定义CNN（可选）

### 🎯 目标检测模型
- **EfficientNetB0Classifier** - 当前默认，种子分类检测，99%+准确率
- **YOLO** - 传统目标检测（可选配置）

## 注意事项

1. **模型文件**: 确保 `weights` 目录下有正确的模型文件（如 FasterNet.pt）
2. **内存使用**: 大型模型可能需要较大内存，请确保系统资源充足
3. **图像格式**: 输入图像应为RGB格式，建议使用高清晰度图像以获得更准确的结果
4. **结果解读**: 模型输出的成分含量是相对值，用于比较不同样本间的差异
5. **环境一致性**: 为确保结果可重现，系统会强制设置随机种子和CUDA参数

## 开发指南

### 添加新模型

1. 在 `models/` 目录下添加新模型的定义文件
2. 在 `models/__init__.py` 中注册新模型
3. 在 `weights/` 目录下放置模型权重文件
4. 使用 `test_model.py list` 命令检查模型是否可用

### 自定义测试

可以修改 `config.py` 中的配置参数，如默认模型、默认图像路径等。

## 许可证

[请在此处添加项目许可证信息]

## 联系方式

[请在此处添加联系方式]
