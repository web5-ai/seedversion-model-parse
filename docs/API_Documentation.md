# 种子成分分析API文档

## 📋 概述

本系统提供基于深度学习的种子图像智能检测和成分分析功能。系统采用YOLO进行目标检测，结合多种CNN模型进行成分分析，可准确预测种子的蛋白质和油脂含量。

**服务地址**: `http://localhost:8123`

## 🚀 API版本

| 版本 | 接口路径 | 特点 | 适用场景 |
|------|----------|------|----------|
| **V1** | `/v1/predict` | 简化结果，核心数据 | 简单应用，快速集成 |
| **V2** | `/v2/predict` | 详细结果，包含检测信息 | 调试监控，完整分析 |

## 📥 请求参数

两个版本使用相同的请求格式：

```json
{
    "image_url": "string",           // 图像路径或URL (必填)
    "model": "FasterNet",            // 分析模型名称 (可选)
    "conf_threshold": 0.9,           // 检测置信度阈值 (可选)
    "iou_threshold": 0.5             // IoU阈值 (可选)
}
```

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `image_url` | string | ✅ | - | 图像URL或本地路径 |
| `model` | string | ❌ | "FasterNet" | 成分分析模型名称 |
| `conf_threshold` | float | ❌ | 0.9 | 检测置信度阈值 (0.0-1.0) |
| `iou_threshold` | float | ❌ | 0.5 | IoU阈值 (0.0-1.0) |

### 支持的模型

| 模型名称 | 描述 | 推荐场景 |
|----------|------|----------|
| `FasterNet` | 快速轻量级模型 | 实时检测，资源受限环境 |
| `ResNet` | 经典残差网络 | 平衡精度和速度 |
| `EfficientNet` | 高效网络 | 高精度要求 |
| `MPViT` | 多尺度视觉Transformer | 复杂场景 |
| `Swin` | Swin Transformer | 最高精度要求 |
| `VanillaNet` | 简化网络 | 快速推理 |

### 参数详解

#### conf_threshold (置信度阈值)
- **作用**: 控制目标检测的严格程度
- **范围**: 0.0 - 1.0
- **效果**:
  - `0.15` (宽松): 检测更多对象，可能包含误检
  - `0.5` (中等): 平衡的检测效果
  - `0.9` (默认): 严格检测，减少误检

#### iou_threshold (IoU阈值)
- **作用**: 控制重复检测的过滤程度 (非极大值抑制)
- **范围**: 0.0 - 1.0
- **效果**:
  - `0.3` (积极过滤): 更积极地删除重叠检测框
  - `0.5` (默认): 平衡的重叠容忍度
  - `0.6` (宽松过滤): 允许更多重叠检测框

## 📤 响应格式对比

### V1 API 响应 (简化版)

```json
{
    "detected": true,                // 是否检测到种子对象
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
    "detected": true,                // 是否检测到种子对象
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

### 字段对比表

| 字段 | V1 | V2 | 类型 | 说明 |
|------|----|----|------|------|
| `success` | ❌ | ✅ | bool | 操作是否成功 |
| `detected` | ✅ | ✅ | bool | 是否检测到对象 |
| `message` | ✅ | ✅ | string | 状态消息 |
| `objects` | ❌ | ✅ | array | 检测对象详情 |
| `protein` | ✅ | ✅ | float | 蛋白质含量 |
| `oil` | ✅ | ✅ | float | 油脂含量 |
| `time_delta` | ✅ | ✅ | float | 总耗时 |

---

## 📡 API 接口详情

### 1. V1 预测接口

**接口**: `POST /v1/predict`
**功能**: 返回简化的检测和分析结果，适用于快速获取核心数据

### 2. V2 预测接口

**接口**: `POST /v2/predict`
**功能**: 返回详细的检测和分析结果，包含检测对象信息和成功状态

---

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
print(f"V1结果: 检测到={result_v1['detected']}, 蛋白质={result_v1['protein']:.1f}%")

# V2 API调用
response_v2 = requests.post("http://localhost:8123/v2/predict", json=data)
result_v2 = response_v2.json()
print(f"V2结果: 成功={result_v2['success']}, 对象数={len(result_v2['objects'])}")
```

## 🔧 特殊功能

### 置信度自动调整
当 `conf_threshold` 设置为 `0.5` 时，系统会自动调整为 `0.9`：
```
⚠️ 后台修改可能比较慢，所以只要检测到后台传递了默认值目前就自动做调整。检测到置信度阈值为0.5，自动调整为0.9
```

### 参数日志输出
每次请求都会输出详细的参数信息：
```
🔍 预测请求参数 - 模型: FasterNet, 置信度阈值: 0.9, IoU阈值: 0.45
🎯 目标检测参数 - 置信度阈值: 0.9, IoU阈值: 0.45
🚀 YOLO检测参数 - 置信度: 0.9, IoU: 0.45
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

## 📈 性能说明

- 两个接口性能完全相同
- 平均响应时间: 1-3秒 (取决于图像大小和检测对象数量)
- 支持并发请求
- 自动CUDA/CPU设备选择

## 🚨 错误处理

### 常见错误响应

```json
{
  "success": false,
  "detected": false,
  "message": "图像获取失败: 文件不存在",
  "objects": [],
  "protein": 0.0,
  "oil": 0.0,
  "time_delta": 0.0
}
```

---

**更新时间**: 2025-08-26
**版本**: V2.0
**维护者**: 种子成分分析系统团队
