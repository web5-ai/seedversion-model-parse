#!/usr/bin/env python3
"""
测试检测和评估API的脚本
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
sys.path.append('.')

def test_detect_objects():
    """测试目标检测方法"""
    print("=" * 60)
    print("测试目标检测方法...")
    print("=" * 60)
    
    try:
        from backend.model_api import ModelAPI
        from PIL import Image
        
        # 初始化ModelAPI
        print("1. 初始化ModelAPI...")
        model_api = ModelAPI('cuda')
        print("   ModelAPI初始化完成")
        
        # 加载测试图像
        print("\n2. 加载测试图像...")
        image_path = 'tests/test_images/image_custom.png'
        if not os.path.exists(image_path):
            print(f"   错误: 测试图像不存在 {image_path}")
            return False
            
        image = Image.open(image_path)
        print(f"   测试图像加载完成: {image.size}")
        
        # 进行目标检测
        print("\n3. 开始目标检测...")
        result = model_api.detect_objects(image)
        
        print(f"   检测结果:")
        print(f"   - 成功: {result['success']}")
        print(f"   - 检测到对象: {result['detected']}")
        print(f"   - 对象数量: {result.get('detection_count', 0)}")
        print(f"   - 耗时: {result.get('time_delta', 0):.4f} 秒")
        
        if result['detected'] and result['objects']:
            print(f"   - 检测详情:")
            for i, obj in enumerate(result['objects']):
                print(f"     对象 {i+1}: 置信度={obj.get('confidence', 0):.3f}, 类别={obj.get('class_name', 'unknown')}")
        
        return result['success']
        
    except Exception as e:
        print(f"目标检测测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_detect_and_eval():
    """测试检测和评估组合方法"""
    print("\n" + "=" * 60)
    print("测试检测和评估组合方法...")
    print("=" * 60)
    
    try:
        from backend.model_api import ModelAPI
        from PIL import Image
        
        # 初始化ModelAPI
        print("1. 初始化ModelAPI...")
        model_api = ModelAPI('cuda')
        print("   ModelAPI初始化完成")
        
        # 加载测试图像
        print("\n2. 加载测试图像...")
        image_path = 'tests/test_images/image_custom.png'
        if not os.path.exists(image_path):
            print(f"   错误: 测试图像不存在 {image_path}")
            return False
            
        image = Image.open(image_path)
        print(f"   测试图像加载完成: {image.size}")
        
        # 进行检测和评估
        print("\n3. 开始检测和评估...")
        result = model_api.detect_and_eval(image, 'FasterNet')
        
        print(f"   组合结果:")
        print(f"   - 成功: {result['success']}")
        print(f"   - 阶段: {result['stage']}")
        print(f"   - 消息: {result['message']}")
        print(f"   - 检测到对象: {result['detected']}")
        print(f"   - 总耗时: {result.get('total_time_delta', 0):.4f} 秒")
        
        if result['detection_result']:
            det_result = result['detection_result']
            print(f"   - 检测结果: 检测到 {det_result.get('detection_count', 0)} 个对象")
        
        if result['evaluation_result']:
            eval_result = result['evaluation_result']
            print(f"   - 评估结果:")
            print(f"     蛋白质: {eval_result.get('protein', 0):.2f}%")
            print(f"     油脂: {eval_result.get('oil', 0):.2f}%")
        
        return result['success']
        
    except Exception as e:
        print(f"检测和评估测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_api_data_models():
    """测试API数据模型"""
    print("\n" + "=" * 60)
    print("测试API数据模型...")
    print("=" * 60)
    
    try:
        from backend.type_cls import DetectAndEvalModel
        
        # 测试默认参数
        print("1. 测试默认参数...")
        model1 = DetectAndEvalModel(img_src="test.jpg")
        print(f"   默认模型: {model1.model_name}")
        print(f"   默认置信度阈值: {model1.conf_threshold}")
        print(f"   默认IoU阈值: {model1.iou_threshold}")
        
        # 测试自定义参数
        print("\n2. 测试自定义参数...")
        model2 = DetectAndEvalModel(
            img_src="test.jpg",
            model_name="ResNet",
            conf_threshold=0.3,
            iou_threshold=0.5
        )
        print(f"   自定义模型: {model2.model_name}")
        print(f"   自定义置信度阈值: {model2.conf_threshold}")
        print(f"   自定义IoU阈值: {model2.iou_threshold}")
        
        print("   ✓ API数据模型测试成功")
        return True
        
    except Exception as e:
        print(f"API数据模型测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_config_integration():
    """测试配置集成"""
    print("\n" + "=" * 60)
    print("测试配置集成...")
    print("=" * 60)
    
    try:
        from config import MODEL_CONFIG
        
        print("检测模型配置:")
        print(f"   detect_model: {MODEL_CONFIG.get('detect_model')}")
        print(f"   detect_model_path: {MODEL_CONFIG.get('detect_model_path')}")
        print(f"   detect_conf_threshold: {MODEL_CONFIG.get('detect_conf_threshold')}")
        print(f"   detect_iou_threshold: {MODEL_CONFIG.get('detect_iou_threshold')}")
        
        # 验证配置完整性
        required_keys = ['detect_model', 'detect_model_path', 'detect_conf_threshold', 'detect_iou_threshold']
        missing_keys = [key for key in required_keys if key not in MODEL_CONFIG]
        
        if missing_keys:
            print(f"   ✗ 缺少配置项: {missing_keys}")
            return False
        else:
            print("   ✓ 配置完整性检查通过")
            return True
        
    except Exception as e:
        print(f"配置集成测试失败: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始检测和评估API测试")
    print("=" * 80)
    
    # 运行所有测试
    tests = [
        ("配置集成", test_config_integration),
        ("API数据模型", test_api_data_models),
        ("目标检测方法", test_detect_objects),
        ("检测和评估组合", test_detect_and_eval)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{test_name}测试出现异常: {str(e)}")
            results[test_name] = False
    
    # 输出测试总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\n总计: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
