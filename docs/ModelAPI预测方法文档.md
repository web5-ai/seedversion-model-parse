# ModelAPI预测方法文档

## 概述

ModelAPI现在提供了多种预测方法，满足不同的使用需求：

1. **predict_v1()** - 返回简化结果，只包含是否检测到和数值预测数据
2. **predict_v2()** - 返回完整结果，包含检测结果详情和数据
3. **draw_detection_boxes()** - 在图片上绘制检测框
4. **predict_with_visualization()** - 预测并可视化检测结果

## 方法详解

### 1. predict_v1() - V1预测方法

**功能**: 返回简化的预测结果，适合只需要基本信息的场景

```python
def predict_v1(self, image, model_name='FasterNet', conf_threshold=None, iou_threshold=None) -> dict:
```

**参数**:
- `image`: 输入图像 (PIL Image)
- `model_name`: 成分分析模型名称，默认'FasterNet'
- `conf_threshold`: 检测置信度阈值，None表示使用配置默认值
- `iou_threshold`: IoU阈值，None表示使用配置默认值

**返回格式**:
```json
{
    "detected": false,           // 是否检测到种子对象
    "protein": 0.0,             // 蛋白质含量（%）
    "oil": 0.0,                 // 油脂含量（%）
    "message": "未检测到种子对象", // 状态消息
    "time_delta": 0.6551        // 总耗时（秒）
}
```

**使用示例**:
```python
from backend.model_api import ModelAPI
from PIL import Image

model_api = ModelAPI('cuda')
image = Image.open('seed_image.jpg')
result = model_api.predict_v1(image, 'ResNet')

if result['detected']:
    print(f"蛋白质: {result['protein']:.2f}%")
    print(f"油脂: {result['oil']:.2f}%")
else:
    print("未检测到种子对象")
```

### 2. predict_v2() - V2预测方法

**功能**: 返回完整的预测结果，包含详细的检测和分析信息

```python
def predict_v2(self, image, model_name='FasterNet', conf_threshold=None, iou_threshold=None) -> dict:
```

**参数**: 与predict_v1相同

**返回格式**:
```json
{
    "success": true,
    "stage": "detection_only",
    "message": "未检测到种子对象",
    "detected": false,
    "detection_result": {
        "success": true,
        "detected": false,
        "objects": [],
        "detection_count": 0,
        "time_delta": 0.1051,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45
    },
    "evaluation_result": null,
    "total_time_delta": 0.1051
}
```

**使用示例**:
```python
result = model_api.predict_v2(image, 'FasterNet')

if result['success']:
    if result['detected']:
        eval_result = result['evaluation_result']
        print(f"蛋白质: {eval_result['protein']:.2f}%")
        print(f"检测到 {result['detection_result']['detection_count']} 个对象")
    else:
        print("未检测到种子对象")
else:
    print(f"预测失败: {result.get('error', '未知错误')}")
```

### 3. draw_detection_boxes() - 检测框绘制方法

**功能**: 在图像上绘制检测框，支持可选的置信度显示

```python
def draw_detection_boxes(self, image, conf_threshold=None, iou_threshold=None, 
                        show_confidence=False, box_color=(0, 255, 0), 
                        text_color=(255, 255, 255), thickness=2):
```

**参数**:
- `image`: 输入图像
- `conf_threshold`: 检测置信度阈值
- `iou_threshold`: IoU阈值
- `show_confidence`: 是否显示置信度，默认False
- `box_color`: 检测框颜色 (R, G, B)，默认绿色
- `text_color`: 文字颜色 (R, G, B)，默认白色
- `thickness`: 线条粗细，默认2

**返回格式**:
```json
{
    "success": true,
    "detected": false,
    "annotated_image": "<PIL.Image对象>",
    "detection_count": 0,
    "detection_details": [],
    "time_delta": 0.1146
}
```

**使用示例**:
```python
# 不显示置信度
result = model_api.draw_detection_boxes(image, show_confidence=False)
if result['success']:
    result['annotated_image'].save('output_no_conf.png')

# 显示置信度，自定义颜色
result = model_api.draw_detection_boxes(
    image, 
    show_confidence=True,
    box_color=(255, 0, 0),  # 红色框
    text_color=(255, 255, 0)  # 黄色文字
)
if result['success']:
    result['annotated_image'].save('output_with_conf.png')
```

### 4. predict_with_visualization() - 预测和可视化组合方法

**功能**: 同时进行预测和可视化，一次调用获得预测结果和标注图像

```python
def predict_with_visualization(self, image, model_name='FasterNet',
                             conf_threshold=None, iou_threshold=None, 
                             show_confidence=False, return_v1_format=True):
```

**参数**:
- `image`: 输入图像
- `model_name`: 成分分析模型名称
- `conf_threshold`: 检测置信度阈值
- `iou_threshold`: IoU阈值
- `show_confidence`: 是否在检测框中显示置信度
- `return_v1_format`: 是否返回v1格式的简化结果，False则返回v2格式

**返回格式**:
```json
{
    "prediction": {
        "detected": false,
        "protein": 0.0,
        "oil": 0.0,
        "message": "未检测到种子对象",
        "time_delta": 0.6551
    },
    "visualization": {
        "success": true,
        "annotated_image": "<PIL.Image对象>",
        "detection_count": 0,
        "detection_details": []
    },
    "total_time_delta": 0.2406,
    "show_confidence": false,
    "format_version": "v1"
}
```

**使用示例**:
```python
# V1格式 + 可视化
result = model_api.predict_with_visualization(
    image, 'FasterNet', 
    show_confidence=False, 
    return_v1_format=True
)

# 获取预测结果
prediction = result['prediction']
if prediction['detected']:
    print(f"蛋白质: {prediction['protein']:.2f}%")
    print(f"油脂: {prediction['oil']:.2f}%")

# 保存标注图像
visualization = result['visualization']
if visualization['success']:
    visualization['annotated_image'].save('result_v1.png')

# V2格式 + 可视化 + 显示置信度
result = model_api.predict_with_visualization(
    image, 'ResNet', 
    show_confidence=True, 
    return_v1_format=False
)

# V2格式的预测结果更详细
prediction = result['prediction']
if prediction['success'] and prediction['detected']:
    eval_result = prediction['evaluation_result']
    print(f"详细分析结果: {eval_result}")

# 保存带置信度的标注图像
visualization = result['visualization']
if visualization['success']:
    visualization['annotated_image'].save('result_v2_with_conf.png')
```

## 方法对比

| 方法 | 返回格式 | 包含检测详情 | 包含标注图像 | 适用场景 |
|------|----------|-------------|-------------|----------|
| predict_v1 | 简化 | ❌ | ❌ | 只需要基本预测结果 |
| predict_v2 | 完整 | ✅ | ❌ | 需要详细检测信息 |
| draw_detection_boxes | 可视化 | ✅ | ✅ | 只需要可视化结果 |
| predict_with_visualization | 组合 | ✅ | ✅ | 需要预测+可视化 |

## 配置选项

### 检测参数
- `conf_threshold`: 置信度阈值，默认0.25
- `iou_threshold`: IoU阈值，默认0.45

### 可视化参数
- `show_confidence`: 是否显示置信度
- `box_color`: 检测框颜色
- `text_color`: 文字颜色
- `thickness`: 线条粗细

### 模型选择
支持的成分分析模型：
- `MPViT`
- `ResNet`
- `FasterNet` (默认)
- `EfficientNet`
- `Swin`
- `VanillaNet`

## 性能特点

1. **智能流程**: 只有检测到种子对象才进行成分分析
2. **灵活格式**: 支持v1简化格式和v2完整格式
3. **可视化选项**: 支持可选的置信度显示
4. **自动管理**: 自动处理模型加载和卸载
5. **错误处理**: 完善的异常处理和状态反馈

## 测试验证

运行测试脚本验证所有方法：

```bash
python test_predict_methods.py
```

测试结果会生成以下文件：
- `test_output_no_conf.png` - 不显示置信度的检测框
- `test_output_with_conf.png` - 显示置信度的检测框
- `test_v1_visualization.png` - V1格式的可视化结果
- `test_v2_visualization.png` - V2格式的可视化结果

## 总结

新的预测方法提供了完整的解决方案：

1. **predict_v1**: 简单快速，适合基础应用
2. **predict_v2**: 详细完整，适合高级应用
3. **draw_detection_boxes**: 专业可视化，支持自定义样式
4. **predict_with_visualization**: 一站式解决方案，预测+可视化

这些方法满足了从简单预测到复杂可视化的各种需求，为不同的应用场景提供了灵活的选择！
