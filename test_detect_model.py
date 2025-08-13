#!/usr/bin/env python3
"""
检测模型测试脚本
"""

import sys
import os
import time
import traceback

# 添加项目根目录到Python路径
sys.path.append('.')

from utils.model_loader import ModelLoader
from PIL import Image
from config import MODEL_CONFIG

def test_yolo_detection():
    """测试YOLO检测模型"""
    print("=" * 60)
    print("开始YOLO检测模型测试...")
    print("=" * 60)
    
    try:
        # 初始化ModelLoader
        print("1. 初始化ModelLoader...")
        loader = ModelLoader(device='cuda')
        print("   ModelLoader初始化完成")
        
        # 加载YOLO检测模型
        print("\n2. 加载YOLO检测模型...")
        success = loader.load_detect_model("YOLO")
        if not success:
            print("   ✗ YOLO模型加载失败")
            return False
        print("   ✓ YOLO模型加载成功")
        
        # 加载测试图像
        print("\n3. 加载测试图像...")
        image_path = 'tests/test_images/image_custom.png'
        if not os.path.exists(image_path):
            print(f"   错误: 测试图像不存在 {image_path}")
            return False
            
        image = Image.open(image_path)
        print(f"   测试图像加载完成: {image.size}")
        
        # 进行目标检测
        print("\n4. 开始目标检测...")
        start_time = time.time()
        
        try:
            results = loader.detect(
                image, 
                conf_threshold=MODEL_CONFIG["detect_conf_threshold"],
                iou_threshold=MODEL_CONFIG["detect_iou_threshold"]
            )
            end_time = time.time()
            
            print(f"   检测完成，耗时: {end_time - start_time:.4f} 秒")
            print(f"   检测结果类型: {type(results)}")
            
            # 分析检测结果
            if results:
                print(f"   检测到 {len(results)} 个结果")
                for i, result in enumerate(results):
                    print(f"   结果 {i+1}: {type(result)}")
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        print(f"     检测框数量: {len(result.boxes)}")
                    if hasattr(result, 'names'):
                        print(f"     类别名称: {result.names}")
            else:
                print("   未检测到任何目标")
            
            print("   ✓ YOLO检测成功")
            
        except Exception as e:
            print(f"   ✗ YOLO检测失败: {str(e)}")
            traceback.print_exc()
            return False
        
        print("\n" + "=" * 60)
        print("YOLO检测模型测试完成!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {str(e)}")
        traceback.print_exc()
        return False

def test_model_info():
    """测试模型信息获取"""
    print("测试模型信息获取...")
    
    try:
        loader = ModelLoader(device='cuda')
        
        # 测试加载不同类型的模型
        print("\n1. 测试回归模型加载...")
        try:
            success = loader.load_model("ResNet")
            if success:
                print("   ✓ ResNet回归模型加载成功")
            else:
                print("   ✗ ResNet回归模型加载失败")
        except Exception as e:
            print(f"   ✗ ResNet回归模型加载失败: {str(e)}")
            traceback.print_exc()
        
        print("\n2. 测试检测模型加载...")
        success = loader.load_detect_model("YOLO")
        if success:
            print("   ✓ YOLO检测模型加载成功")
        else:
            print("   ✗ YOLO检测模型加载失败")
        
        return True
        
    except Exception as e:
        print(f"模型信息测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_model_switching():
    """测试模型切换功能"""
    print("测试模型切换功能...")
    
    try:
        loader = ModelLoader(device='cuda')
        
        # 先加载回归模型
        print("1. 加载回归模型...")
        loader.load_model("ResNet")
        print("   当前模型:", getattr(loader, 'model_name', 'Unknown'))
        
        # 切换到检测模型
        print("2. 切换到检测模型...")
        loader.load_detect_model("YOLO")
        print("   当前模型:", getattr(loader, 'model_name', 'Unknown'))
        
        # 再切换回回归模型
        print("3. 切换回回归模型...")
        loader.load_model("FasterNet")
        print("   当前模型:", getattr(loader, 'model_name', 'Unknown'))
        
        print("✓ 模型切换测试成功")
        return True
        
    except Exception as e:
        print(f"模型切换测试失败: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "info":
            test_model_info()
        elif test_type == "switch":
            test_model_switching()
        else:
            test_yolo_detection()
    else:
        test_yolo_detection()
