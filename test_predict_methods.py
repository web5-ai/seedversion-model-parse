#!/usr/bin/env python3
"""
测试ModelAPI新增的预测方法
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
sys.path.append('.')

def test_predict_v1():
    """测试V1预测方法（简化结果）"""
    print("=" * 60)
    print("测试V1预测方法...")
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
        
        # 进行V1预测
        print("\n3. 开始V1预测...")
        result = model_api.predict_v1(image, 'FasterNet')
        
        print(f"   V1预测结果:")
        print(f"   - 检测到对象: {result['detected']}")
        print(f"   - 蛋白质: {result['protein']:.2f}%")
        print(f"   - 油脂: {result['oil']:.2f}%")
        print(f"   - 消息: {result['message']}")
        print(f"   - 耗时: {result['time_delta']:.4f} 秒")
        
        return True
        
    except Exception as e:
        print(f"V1预测测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_predict_v2():
    """测试V2预测方法（完整结果）"""
    print("\n" + "=" * 60)
    print("测试V2预测方法...")
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
        
        # 进行V2预测
        print("\n3. 开始V2预测...")
        result = model_api.predict_v2(image, 'FasterNet')
        
        print(f"   V2预测结果:")
        print(f"   - 成功: {result.get('success', False)}")
        print(f"   - 阶段: {result.get('stage', 'unknown')}")
        print(f"   - 检测到对象: {result.get('detected', False)}")
        print(f"   - 总耗时: {result.get('total_time_delta', 0):.4f} 秒")
        
        if result.get('detection_result'):
            det_result = result['detection_result']
            print(f"   - 检测详情: 检测到 {det_result.get('detection_count', 0)} 个对象")
        
        if result.get('evaluation_result'):
            eval_result = result['evaluation_result']
            print(f"   - 评估详情:")
            print(f"     蛋白质: {eval_result.get('protein', 0):.2f}%")
            print(f"     油脂: {eval_result.get('oil', 0):.2f}%")
        
        return True
        
    except Exception as e:
        print(f"V2预测测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_draw_detection_boxes():
    """测试检测框绘制方法"""
    print("\n" + "=" * 60)
    print("测试检测框绘制方法...")
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
        
        # 测试不显示置信度
        print("\n3. 测试绘制检测框（不显示置信度）...")
        result1 = model_api.draw_detection_boxes(image, show_confidence=False)
        
        print(f"   绘制结果:")
        print(f"   - 成功: {result1['success']}")
        print(f"   - 检测到对象: {result1['detected']}")
        print(f"   - 检测数量: {result1['detection_count']}")
        print(f"   - 耗时: {result1['time_delta']:.4f} 秒")
        
        if result1['success'] and result1['annotated_image']:
            # 保存标注图像
            output_path1 = 'test_output_no_conf.png'
            result1['annotated_image'].save(output_path1)
            print(f"   - 标注图像已保存: {output_path1}")
        
        # 测试显示置信度
        print("\n4. 测试绘制检测框（显示置信度）...")
        result2 = model_api.draw_detection_boxes(image, show_confidence=True)
        
        print(f"   绘制结果:")
        print(f"   - 成功: {result2['success']}")
        print(f"   - 检测到对象: {result2['detected']}")
        print(f"   - 检测数量: {result2['detection_count']}")
        print(f"   - 耗时: {result2['time_delta']:.4f} 秒")
        
        if result2['success'] and result2['annotated_image']:
            # 保存标注图像
            output_path2 = 'test_output_with_conf.png'
            result2['annotated_image'].save(output_path2)
            print(f"   - 标注图像已保存: {output_path2}")
        
        return True
        
    except Exception as e:
        print(f"检测框绘制测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_predict_with_visualization():
    """测试预测和可视化组合方法"""
    print("\n" + "=" * 60)
    print("测试预测和可视化组合方法...")
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
        
        # 测试V1格式 + 可视化
        print("\n3. 测试V1格式 + 可视化...")
        result1 = model_api.predict_with_visualization(
            image, 'FasterNet', 
            show_confidence=False, 
            return_v1_format=True
        )
        
        print(f"   V1组合结果:")
        print(f"   - 格式版本: {result1['format_version']}")
        print(f"   - 总耗时: {result1['total_time_delta']:.4f} 秒")
        print(f"   - 预测结果:")
        pred1 = result1['prediction']
        print(f"     检测到: {pred1['detected']}")
        print(f"     蛋白质: {pred1['protein']:.2f}%")
        print(f"     油脂: {pred1['oil']:.2f}%")
        print(f"   - 可视化结果:")
        vis1 = result1['visualization']
        print(f"     成功: {vis1['success']}")
        print(f"     检测数量: {vis1['detection_count']}")
        
        if vis1['success'] and vis1['annotated_image']:
            output_path1 = 'test_v1_visualization.png'
            vis1['annotated_image'].save(output_path1)
            print(f"     标注图像已保存: {output_path1}")
        
        # 测试V2格式 + 可视化 + 显示置信度
        print("\n4. 测试V2格式 + 可视化 + 显示置信度...")
        result2 = model_api.predict_with_visualization(
            image, 'FasterNet', 
            show_confidence=True, 
            return_v1_format=False
        )
        
        print(f"   V2组合结果:")
        print(f"   - 格式版本: {result2['format_version']}")
        print(f"   - 显示置信度: {result2['show_confidence']}")
        print(f"   - 总耗时: {result2['total_time_delta']:.4f} 秒")
        print(f"   - 预测结果:")
        pred2 = result2['prediction']
        print(f"     成功: {pred2.get('success', False)}")
        print(f"     检测到: {pred2.get('detected', False)}")
        print(f"   - 可视化结果:")
        vis2 = result2['visualization']
        print(f"     成功: {vis2['success']}")
        print(f"     检测数量: {vis2['detection_count']}")
        
        if vis2['success'] and vis2['annotated_image']:
            output_path2 = 'test_v2_visualization.png'
            vis2['annotated_image'].save(output_path2)
            print(f"     标注图像已保存: {output_path2}")
        
        return True
        
    except Exception as e:
        print(f"预测和可视化组合测试失败: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始ModelAPI预测方法测试")
    print("=" * 80)
    
    # 运行所有测试
    tests = [
        ("V1预测方法", test_predict_v1),
        ("V2预测方法", test_predict_v2),
        ("检测框绘制", test_draw_detection_boxes),
        ("预测和可视化组合", test_predict_with_visualization)
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
        print("\n生成的测试文件:")
        test_files = [
            'test_output_no_conf.png',
            'test_output_with_conf.png', 
            'test_v1_visualization.png',
            'test_v2_visualization.png'
        ]
        for file in test_files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (未生成)")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
