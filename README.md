# 油菜籽成分分析系统

## 分支维护：QG-rise

模型调试问题：计算滑动窗口分块时，出现除不尽的情况，导致张量再划分报错
pip install torch-2.6.0+cu126-cp310-abi3-win_amd64.whl
swin模型，训练方式，输入图像是256x256，窗口大小8x8

## 项目背景

本项目是一个基于深度学习的油菜籽成分分析系统，旨在通过图像识别技术快速分析油菜籽中的主要成分含量，包括油酸、亚油酸、亚麻酸、棕榈酸和硬脂酸等。该系统可以帮助农业工作者和食品加工企业快速评估油菜籽的品质，提高生产效率和产品质量。

## 运行环境

### 系统要求

- Python 3.8+
- 依赖库详见requirements.txt

### 硬件要求

- CPU: 支持现代指令集的多核处理器
- 内存: 至少4GB RAM
- GPU: 可选，但推荐使用NVIDIA GPU以加速模型推理
- 存储: 至少1GB可用空间

## 快速运行

1. 克隆仓库到本地

```bash
git clone <仓库地址>
cd mendianyunying/pythonVesion
```

2. 设置环境（自动创建虚拟环境并安装依赖）

```bash
chmod +x setup.sh
./setup.sh
```

3. 激活虚拟环境

```bash
source venv/bin/activate
```

4. 运行测试脚本

```bash
python test_model.py
```

5. 使用自定义图像进行测试

```bash
python test_model.py --image path/to/your/image.jpg
```

## 开发和调试

### 项目结构

```
mendianyunying/pythonVesion/
├── config.py                 # 配置文件
├── backend/                  # 后端接口
│   ├── main.py               # fastapi运行入口
│   └── run.py                # 项目运行配置，由main.py调用
│   └── model_api.py          # 模型api，从model_loader封装，直接提供预测接口
│   └── tools.py              # 后台用到的工具函数都在这，包括下载图像、存储等
│   └── type_cls.py           # 定义接口参数的类
│   └── start_service.bat     # 项目一键运行脚本，根据需求和环境进行配置
├── models/                   # 模型定义
│   ├── __init__.py           # 初始化，定义了对外提供的模型接口
│   └── model_zoo.py          # 客户的模型定义文件，包含了不同模型的定义
|   └── build_mpvit.py        # 客户定义的MPViT模型
|   └── build_vanillanet.py   # 客户定义的Vanillanet模型
|   └── build_swinv2.py       # 客户定义的Swin模型
|   └── swinv2_config.py      # 客户定义的Swin模型配置文件
|   └── swinv2_config_large.py# 客户定义的Swin模型配置文件
|   └── 其他模型文件           # 根据客户提供的模型封装了__init__.py中的接口
├── utils/                    # 工具函数
│   ├── environment.py        # 环境检查
│   └── image_processor.py    # 图像处理
│   └── model_loader.py       # 模型加载器，从models里加载模型，封装了预处理、预测等功能
├── tests/                    # 测试文件
│   └── test_images/          # 测试图像
├── weights/                  # 预训练模型权重，不上传
├── results/                  # 结果输出目录
├── requirements.txt          # 依赖库列表
├── setup.sh                  # 环境设置脚本
├── README.md                 # 项目说明文档
├── unzip.sh                  # 解压脚本，项目上传到服务器部署时用于解压和依赖下载的脚本
├── zip.ps1                   # 训练脚本，将项目打包上传到服务器的脚本
└── test_model.py             # 测试脚本，包含自动化测试auto_model_test()，可以自动使用测试图像测试所有模型
```

### 调试模式&环境检查

可以通过在 `test_model.py`中添加 `--debug`参数启用调试模式，不推荐使用：
建议查看test_model下main里的测试函数，调整为auto_model_test()，可以自动测试所有模型以及环境配置。

```bash
python test_model.py
```

## 关键技术

1. **深度学习模型**: 使用各类预训练的模型，进行油菜籽的油脂和蛋白质含量预测。
2. **模型加载机制**: 实现了灵活的模型加载器，可以自动识别不同格式的模型文件并加载。
3. **图像预处理**: 标准化的图像预处理流程，确保输入模型的图像具有一致的格式和特征分布。

## 当前模型信息

运行后会在 `/weights`目录下生成各个模型的 `state_dict_info` 和 `model_info`文件，包含模型结构和状态字典的信息，如果一致说明模型完全加载（不一致会报错，加载时采用严格模式）。

## 入门必备基础

要充分理解和开发本项目，建议具备以下基础知识：

1. **Python编程**: 熟悉Python基本语法和常用库
2. **深度学习基础**: 了解卷积神经网络(CNN)的基本原理
3. **PyTorch框架**: 熟悉PyTorch的基本用法
4. **图像处理**: 了解基本的图像处理技术
5. **农业知识**: 对油菜籽成分及其意义有基本了解

## 参考资料

1. [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
2. [ResNet论文](https://arxiv.org/abs/1512.03385)
3. [图像分类教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
4. [迁移学习指南](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 注意事项

1. **模型文件**: 确保 `weights`目录下有正确的模型文件（fasternet_model.pt）。
2. **内存使用**: 模型较大，请确保系统有足够的内存。
3. **图像格式**: 输入图像应为RGB格式，建议使用高清晰度的油菜籽图像以获得更准确的结果。
4. **结果解读**: 模型输出的成分含量是相对值，用于比较不同样本间的差异，不代表绝对含量百分比。
5. **模型更新**: 如需使用新的模型，请确保模型输出维度与配置文件中的成分数量一致。
6. **环境兼容性**: 本项目在Python 3.8-3.13环境下测试通过，其他版本可能需要调整依赖库版本。
7. **GPU加速**: 如有NVIDIA GPU，建议启用GPU加速以提高处理速度。

## 许可证

[请在此处添加项目许可证信息]

## 联系方式

[loop]
