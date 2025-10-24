#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
种子识别模型使用示例

演示如何使用模型结构 + 权重文件进行推理
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from inference import load_model

def main():
    """使用示例"""
    
    print("🌱 精简版种子识别模型使用示例")
    print("=" * 50)
    
    # 模型和权重文件路径
    models = {
        "EfficientNet-B0": {
            "name": "efficientnet_b0",
            "weights": "models/efficientnet_b0_weights.pth"
        },
        "ResNet18": {
            "name": "resnet18", 
            "weights": "models/resnet18_weights.pth"
        }
    }
    
    # 测试图片
    test_image = "../example/test.jpg"
    if not Path(test_image).exists():
        print(f"⚠️ 测试图片不存在: {test_image}")
        print("请将图片路径替换为实际存在的图片")
        return
    
    # 测试每个模型
    for model_display_name, model_config in models.items():
        model_name = model_config["name"]
        weights_path = model_config["weights"]
        
        if not Path(weights_path).exists():
            print(f"⚠️ 权重文件不存在: {weights_path}")
            continue
        
        print(f"\n🧠 测试模型: {model_display_name}")
        print("-" * 40)
        
        try:
            # 加载模型
            inference = load_model(model_name, weights_path)
            
            # 获取模型信息
            info = inference.get_model_info()
            print(f"参数量: {info['total_parameters']:,}")
            print(f"模型大小: {info['model_size_mb']:.1f}MB")
            print(f"设备: {info['device']}")
            
            # 预测
            result = inference.predict(test_image)
            
            print(f"预测结果: {result['predicted_class']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"推理时间: {result['inference_time_ms']:.1f}ms")
            print(f"详细概率:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
            
        except Exception as e:
            print(f"❌ 模型测试失败: {e}")
    
    print(f"\n🎉 测试完成!")

def batch_example():
    """批量预测示例"""
    
    print("\n📚 批量预测示例:")
    print("-" * 30)
    
    # 加载推荐模型
    weights_path = "models/efficientnet_b0_weights.pth"
    if not Path(weights_path).exists():
        print(f"⚠️ 权重文件不存在: {weights_path}")
        return
    
    inference = load_model("efficientnet_b0", weights_path)
    
    # 批量预测（这里用同一张图片演示）
    test_image = "../example/test.jpg"
    if Path(test_image).exists():
        image_paths = [test_image] * 3  # 模拟3张图片
        
        results = inference.predict_batch(image_paths, batch_size=2)
        
        for i, result in enumerate(results, 1):
            print(f"  图片{i}: {result['predicted_class']} (置信度: {result['confidence']:.3f})")

if __name__ == "__main__":
    main()
    batch_example()
