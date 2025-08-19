# 油菜籽检测与成分分析 API 文档

## 概述

本API提供油菜籽图像的智能检测和成分分析功能，支持多种深度学习模型，可以检测图像中的油菜籽对象并预测其蛋白质和油脂含量。

**基础URL**: `http://localhost:8123`

## API版本

- **v1**: 简化版本，返回基础检测和分析结果
- **v2**: 完整版本，返回详细的检测信息和分析数据

---

## 🔧 通用参数说明

### 请求参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `image_url` | string | ✅ | - | 图像URL或本地路径 |
| `model` | string | ❌ | "FasterNet" | 成分分析模型名称 |
| `conf_threshold` | float | ❌ | 0.25 | 检测置信度阈值 (0.0-1.0) |
| `iou_threshold` | float | ❌ | 0.45 | IoU阈值 (0.0-1.0) |

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
  - `0.25` (默认): 平衡的检测效果
  - `0.4` (严格): 只保留高置信度检测，减少误检

#### iou_threshold (IoU阈值)
- **作用**: 控制重复检测的过滤程度 (非极大值抑制)
- **范围**: 0.0 - 1.0
- **效果**:
  - `0.3` (积极过滤): 更积极地删除重叠检测框
  - `0.45` (默认): 平衡的重叠容忍度
  - `0.6` (宽松过滤): 允许更多重叠检测框

---

## 📡 API 接口

### 1. V1 预测接口 (简化版)

**接口**: `POST /v1/predict`

**功能**: 返回简化的检测和分析结果，适用于快速获取核心数据

#### 请求示例

```bash
curl -X POST "http://localhost:8123/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/seed_image.jpg",
    "model": "ResNet",
    "conf_threshold": 0.25,
    "iou_threshold": 0.45
  }'
```

#### 响应格式

```json
{
  "success": true,
  "detection_result": {
    "detection_count": 81,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45
  },
  "evaluation_result": {
    "protein": 25.50,
    "oil": 38.87
  },
  "total_time_delta": 4.808
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | boolean | 整体操作是否成功 |
| `detection_result.detection_count` | integer | 检测到的对象数量 |
| `detection_result.conf_threshold` | float | 使用的置信度阈值 |
| `detection_result.iou_threshold` | float | 使用的IoU阈值 |
| `evaluation_result.protein` | float | 蛋白质含量 (%) |
| `evaluation_result.oil` | float | 油脂含量 (%) |
| `total_time_delta` | float | 总处理时间 (秒) |

---

### 2. V2 预测接口 (完整版)

**接口**: `POST /v2/predict`

**功能**: 返回完整的检测和分析结果，包含详细的检测信息

#### 请求示例

```bash
curl -X POST "http://localhost:8123/v2/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/seed_image.jpg",
    "model": "ResNet",
    "conf_threshold": 0.3,
    "iou_threshold": 0.4
  }'
```

#### 响应格式

```json
{
  "success": true,
  "stage": "complete",
  "message": "检测和分析完成",
  "detected": true,
  "detection_result": {
    "success": true,
    "detected": true,
    "objects": [
      {
        "confidence": 0.85,
        "class_id": 0,
        "class_name": "rapeseed",
        "bbox": [100, 150, 300, 350]
      }
    ],
    "detection_count": 81,
    "time_delta": 0.654,
    "conf_threshold": 0.3,
    "iou_threshold": 0.4
  },
  "evaluation_result": {
    "protein": 25.50,
    "oil": 38.87,
    "time_delta": 3.914,
    "memory_cost": 256.5
  },
  "total_time_delta": 4.568,
  "image_hash": "abc123...",
  "model_name": "ResNet"
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | boolean | 整体操作是否成功 |
| `stage` | string | 执行阶段 ("detection_only", "complete", "error") |
| `message` | string | 状态消息 |
| `detected` | boolean | 是否检测到对象 |
| `detection_result` | object | 详细检测结果 |
| `detection_result.objects` | array | 检测到的对象列表 |
| `detection_result.objects[].confidence` | float | 对象置信度 |
| `detection_result.objects[].bbox` | array | 边界框坐标 [x1, y1, x2, y2] |
| `evaluation_result` | object | 成分分析结果 |
| `evaluation_result.protein` | float | 蛋白质含量 (%) |
| `evaluation_result.oil` | float | 油脂含量 (%) |
| `evaluation_result.time_delta` | float | 分析耗时 (秒) |
| `evaluation_result.memory_cost` | float | 内存消耗 (MB) |
| `total_time_delta` | float | 总处理时间 (秒) |
| `image_hash` | string | 图像哈希值 |
| `model_name` | string | 使用的模型名称 |

---

## 🚨 错误处理

### 错误响应格式

```json
{
  "success": false,
  "error": "错误描述",
  "stage": "error",
  "detected": false,
  "total_time_delta": 0.0
}
```

### 常见错误码

| HTTP状态码 | 错误类型 | 说明 |
|------------|----------|------|
| 400 | Bad Request | 请求参数错误 |
| 404 | Not Found | 图像URL无法访问 |
| 422 | Unprocessable Entity | 参数验证失败 |
| 500 | Internal Server Error | 服务器内部错误 |

---

## 📊 使用示例

### Python 示例

```python
import requests
import json

# V1 API 调用
def call_v1_api(image_url, model="ResNet"):
    url = "http://localhost:8123/v1/predict"
    data = {
        "image_url": image_url,
        "model": model,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    
    if result.get("success"):
        print(f"检测到 {result['detection_result']['detection_count']} 个对象")
        print(f"蛋白质: {result['evaluation_result']['protein']:.2f}%")
        print(f"油脂: {result['evaluation_result']['oil']:.2f}%")
    else:
        print(f"检测失败: {result.get('error', '未知错误')}")

# V2 API 调用
def call_v2_api(image_url, model="ResNet"):
    url = "http://localhost:8123/v2/predict"
    data = {
        "image_url": image_url,
        "model": model,
        "conf_threshold": 0.3,
        "iou_threshold": 0.4
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    
    if result.get("success") and result.get("detected"):
        detection = result["detection_result"]
        evaluation = result["evaluation_result"]
        
        print(f"检测结果: {detection['detection_count']} 个对象")
        print(f"平均置信度: {sum(obj['confidence'] for obj in detection['objects']) / len(detection['objects']):.3f}")
        print(f"成分分析: 蛋白质 {evaluation['protein']:.2f}%, 油脂 {evaluation['oil']:.2f}%")
        print(f"总耗时: {result['total_time_delta']:.3f}秒")
    else:
        print(f"检测失败: {result.get('message', '未知错误')}")

# 使用示例
image_url = "https://example.com/rapeseed_image.jpg"
call_v1_api(image_url, "ResNet")
call_v2_api(image_url, "EfficientNet")
```

### JavaScript 示例

```javascript
// V1 API 调用
async function callV1API(imageUrl, model = "ResNet") {
    const response = await fetch("http://localhost:8123/v1/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            image_url: imageUrl,
            model: model,
            conf_threshold: 0.25,
            iou_threshold: 0.45
        })
    });
    
    const result = await response.json();
    
    if (result.success) {
        console.log(`检测到 ${result.detection_result.detection_count} 个对象`);
        console.log(`蛋白质: ${result.evaluation_result.protein.toFixed(2)}%`);
        console.log(`油脂: ${result.evaluation_result.oil.toFixed(2)}%`);
    } else {
        console.log(`检测失败: ${result.error || '未知错误'}`);
    }
}

// 使用示例
callV1API("https://example.com/rapeseed_image.jpg", "ResNet");
```

---

## 🔧 性能优化建议

### 1. 模型选择
- **快速检测**: 使用 `FasterNet` 或 `VanillaNet`
- **平衡性能**: 使用 `ResNet` 或 `EfficientNet`
- **最高精度**: 使用 `Swin` 或 `MPViT`

### 2. 参数调优
- **高密度种子**: `conf_threshold=0.3, iou_threshold=0.3`
- **低质量图像**: `conf_threshold=0.15, iou_threshold=0.5`
- **精确计数**: `conf_threshold=0.4, iou_threshold=0.35`

### 3. 图像要求
- **分辨率**: 建议 640x640 以上
- **格式**: 支持 JPG, PNG, BMP
- **质量**: 清晰度越高，检测效果越好

---

## 📝 更新日志

### v1.0.0 (2025-08-19)
- ✅ 移除响应中的 `seed_info` 字段
- ✅ 统一字段命名 (`model_name` → `model`, `img_src` → `image_url`)
- ✅ 优化API响应结构
- ✅ 完善错误处理机制
