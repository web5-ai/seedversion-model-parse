predict_v1

{
  "image_url": "string",  // 图像文件的URL或本地路径（必需）
  "model": "string",      // 用于成分分析的模型名称，可选值: 'MPViT', 'ResNet', 'FasterNet', 'EfficientNet', 'Swin', 'VanillaNet'，默认为'FasterNet'
  "conf_threshold": float, // 检测置信度阈值，None表示使用配置默认值
  "iou_threshold": float   // IoU阈值，None表示使用配置默认值
}

返回值
{
  "object_classes_counts": {},        // 检测到的种子类别统计
  "protein": float,                   // 蛋白质含量百分比
  "oil": float,                       // 油脂含量百分比
  "message": "string",               // 状态消息
  "time_delta": float                 // 总耗时（秒）
}

predict_v2

{
  "success": true/false,              // 整体操作是否成功
  "object_classes_counts": {},        // 检测到的种子类别统计
  "message": "string",               // 状态消息
  "objects": [                       // 检测到的对象列表
    {
      "confidence": float,           // 检测置信度
      "class_id": int,               // 类别ID
      "class_name": "string",       // 类别名称
      "bbox": [x1, y1, x2, y2]       // 边界框坐标
    },
    ...                            // 若干相同的对象
  ],
  "protein": float,                   // 蛋白质含量百分比
  "oil": float,                       // 油脂含量百分比
  "time_delta": float                 // 总耗时（秒）
}