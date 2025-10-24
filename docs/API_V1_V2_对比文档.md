# 种子成分分析API文档 - V1 vs V2

## 📋 概述

本系统提供两个预测接口：
- **V1 API**: 简化接口，返回核心结果
- **V2 API**: 增强接口，返回详细信息

两个接口功能完全相同，仅返回数据格式不同。

## 🔗 接口地址

| 版本 | 方法 | 路径          | 描述         |
| ---- | ---- | ------------- | ------------ |
| V1   | POST | `/v1/predict` | 简化结果接口 |
| V2   | POST | `/v2/predict` | 详细结果接口 |

## 📥 请求参数

两个接口使用相同的请求参数：

```json
{
    "image_url": "string",           // 图像路径或URL (必填)
    "model": "FasterNet",            // 分析模型名称 (可选，默认FasterNet)
    "conf_threshold": 0.9,           // 检测置信度阈值 (可选，默认0.9)
    "iou_threshold": 0.5             // IoU阈值 (可选，默认0.5)
}
```

### 参数说明

- **image_url**: 支持本地路径或网络URL
- **model**: 可选值 `MPViT`, `ResNet`, `FasterNet`, `EfficientNet`, `Swin`, `VanillaNet`
- **conf_threshold**: 范围 0.0-1.0，默认0.9，值为0.5时自动调整为0.9
- **iou_threshold**: 范围 0.0-1.0，默认0.5

## 📤 响应格式对比

### V1 API 响应 (简化版)

```json
{
    "object_classes_counts": true,                // 是否检测到种子对象
    "protein": 45.2,                 // 蛋白质含量 (%)
    "oil": 38.7,                     // 油脂含量 (%)
    "message": "检测和分析完成",      // 状态消息
    "time_delta": 2.35               // 总耗时 (秒)
}
```

### V2 API 响应 (详细版)

```json
{
    "success": true,                 // 操作是否成功
    "object_classes_counts": true,                // 是否检测到种子对象
    "message": "检测和分析完成",      // 状态消息
    "objects": [                     // 检测到的对象列表
        {
            "confidence": 0.95,
            "bbox": [100, 150, 200, 250],
            "class": "seed"
        }
    ],
    "protein": 45.2,                 // 蛋白质含量 (%)
    "oil": 38.7,                     // 油脂含量 (%)
    "time_delta": 2.35               // 总耗时 (秒)
}
```

## 📊 字段对比表

| 字段                    | V1  | V2  | 类型   | 说明           |
| ----------------------- | --- | --- | ------ | -------------- |
| `success`               | ❌   | ✅   | bool   | 操作是否成功   |
| `object_classes_counts` | ✅   | ✅   | bool   | 是否检测到对象 |
| `message`               | ✅   | ✅   | string | 状态消息       |
| `objects`               | ❌   | ✅   | array  | 检测对象详情   |
| `protein`               | ✅   | ✅   | float  | 蛋白质含量     |
| `oil`                   | ✅   | ✅   | float  | 油脂含量       |
| `time_delta`            | ✅   | ✅   | float  | 总耗时         |

## 🔄 处理流程

1. **图像加载**: 下载或读取图像文件
2. **目标检测**: 使用YOLO模型检测种子对象
3. **成分分析**: 对检测到的对象进行成分分析
4. **结果返回**: 根据API版本返回相应格式

## 📝 响应示例

### 成功检测到对象

**V1响应:**
```json
{
    "object_classes_counts": true,
    "protein": 42.8,
    "oil": 35.6,
    "message": "检测和分析完成",
    "time_delta": 1.85
}
```

**V2响应:**
```json
{
    "success": true,
    "object_classes_counts": true,
    "message": "检测和分析完成",
    "objects": [
        {
            "confidence": 0.92,
            "bbox": [120, 180, 220, 280],
            "class": "seed"
        }
    ],
    "protein": 42.8,
    "oil": 35.6,
    "time_delta": 1.85
}
```

### 未检测到对象

**V1响应:**
```json
{
    "object_classes_counts": false,
    "protein": 0.0,
    "oil": 0.0,
    "message": "未检测到种子对象",
    "time_delta": 0.65
}
```

**V2响应:**
```json
{
    "success": true,
    "object_classes_counts": false,
    "message": "未检测到种子对象",
    "objects": [],
    "protein": 0.0,
    "oil": 0.0,
    "time_delta": 0.65
}
```

### 处理失败

**V1响应:**
```json
{
    "object_classes_counts": false,
    "protein": 0.0,
    "oil": 0.0,
    "message": "图像获取失败: 文件不存在",
    "time_delta": 0.0
}
```

**V2响应:**
```json
{
    "success": false,
    "object_classes_counts": false,
    "message": "图像获取失败: 文件不存在",
    "objects": [],
    "protein": 0.0,
    "oil": 0.0,
    "time_delta": 0.0
}
```

## 🎯 使用建议

### 选择V1 API的场景
- 简单的客户端应用
- 只需要最终的分析结果
- 对响应大小有要求
- 快速集成

### 选择V2 API的场景
- 需要检测详情信息
- 需要判断操作是否成功
- 调试和监控应用
- 需要显示检测框位置

## 🔧 特殊功能

### 置信度自动调整
当 `conf_threshold` 设置为 `0.5` 时，系统会自动调整为 `0.9`，并在日志中记录：
```
⚠️ 检测到置信度阈值为0.5，自动调整为0.9
```

### 参数日志输出
每次请求都会输出详细的参数信息：
```
🔍 预测请求参数 - 模型: FasterNet, 置信度阈值: 0.9, IoU阈值: 0.45
🎯 目标检测参数 - 置信度阈值: 0.9, IoU阈值: 0.45
🚀 YOLO检测参数 - 置信度: 0.9, IoU: 0.45
```

## 📈 性能说明

- 两个接口性能完全相同
- 平均响应时间: 1-3秒 (取决于图像大小和检测对象数量)
- 支持并发请求
- 自动CUDA/CPU设备选择

## 🚀 快速开始

### cURL示例

```bash
# V1 API
curl -X POST "http://localhost:8123/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "path/to/image.jpg",
    "model": "FasterNet",
    "conf_threshold": 0.9
  }'

# V2 API  
curl -X POST "http://localhost:8123/v2/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "path/to/image.jpg",
    "model": "FasterNet",
    "conf_threshold": 0.9
  }'
```

### Python示例

```python
import requests

# 请求数据
data = {
    "image_url": "tests/images/sample.jpg",
    "model": "FasterNet",
    "conf_threshold": 0.9,
    "iou_threshold": 0.5
}

# V1 API调用
response_v1 = requests.post("http://localhost:8123/v1/predict", json=data)
result_v1 = response_v1.json()
print(f"V1结果: 检测到={result_v1['object_classes_counts']}, 蛋白质={result_v1['protein']:.1f}%")

# V2 API调用
response_v2 = requests.post("http://localhost:8123/v2/predict", json=data)
result_v2 = response_v2.json()
print(f"V2结果: 成功={result_v2['success']}, 对象数={len(result_v2['objects'])}")
```

---

**更新时间**: 2025-08-26  
**版本**: V1.0  
**维护者**: 种子成分分析系统团队
