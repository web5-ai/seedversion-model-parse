# 🌱 精简版种子识别模型

## 📦 包含内容

这是一个精简的种子识别模型包，只包含必要的组件：

- **模型结构定义** (`src/models.py`)
- **推理接口** (`src/inference.py`) 
- **预训练权重** (`models/*.pth`)
- **使用示例** (`example.py`)

## 📁 文件结构

```
simple_seed_models/
├── src/
│   ├── models.py              # 模型结构定义
│   └── inference.py           # 推理接口
├── models/
│   ├── efficientnet_b0_weights.pth  # EfficientNet-B0权重 (推荐)
│   └── resnet18_weights.pth         # ResNet18权重 (最快)
├── example.py                 # 使用示例
└── README.md                  # 本文档
```

## 🚀 快速使用

### 1. 安装依赖
```bash
pip install torch torchvision Pillow
```

### 2. 基本使用
```python
import sys
sys.path.append('simple_seed_models/src')

from inference import load_model

# 加载模型 (推荐EfficientNet-B0)
inference = load_model(
    model_name="efficientnet_b0",
    weights_path="simple_seed_models/models/efficientnet_b0_weights.pth"
)

# 预测图片
result = inference.predict("your_image.jpg")

print(f"预测: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.3f}")
```

### 3. 运行示例
```bash
cd simple_seed_models
python example.py
```

## 🎯 模型选择

| 模型 | 权重文件 | 特点 |
|------|----------|------|
| **EfficientNet-B0** | `efficientnet_b0_weights.pth` | **推荐** - 100%准确率, 15.3MB |
| **ResNet18** | `resnet18_weights.pth` | 最快速度 - 99.38%准确率, ~7ms |

## 💡 核心优势

1. **极简设计** - 只有2个核心文件 + 权重
2. **独立运行** - 不依赖复杂的配置文件
3. **易于集成** - 直接复制到任何项目中使用
4. **高性能** - 99%+的识别准确率

## 🔧 自定义使用

### 加载不同模型
```python
# EfficientNet-B0 (推荐)
inference = load_model("efficientnet_b0", "models/efficientnet_b0_weights.pth")

# ResNet18 (最快)
inference = load_model("resnet18", "models/resnet18_weights.pth")
```

### 批量预测
```python
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = inference.predict_batch(image_paths, batch_size=8)

for result in results:
    print(f"{result['image_path']}: {result['predicted_class']}")
```

### 获取模型信息
```python
info = inference.get_model_info()
print(f"参数量: {info['total_parameters']:,}")
print(f"模型大小: {info['model_size_mb']:.1f}MB")
```

## 📊 识别类别

- **background**: 背景/无种子
- **rapeseed**: 油菜籽

## 🎉 就是这么简单！

只需要3步：
1. 复制文件到你的项目
2. 安装PyTorch依赖
3. 导入并使用

没有复杂的配置，没有多余的文件，开箱即用！🚀
