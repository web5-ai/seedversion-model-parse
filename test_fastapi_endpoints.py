#!/usr/bin/env python3
"""
测试FastAPI端点的脚本
"""

import requests
import json
import time
import os

# API基础URL
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """测试根端点"""
    print("=" * 60)
    print("测试根端点...")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"根端点测试失败: {str(e)}")
        return False

def test_predict_endpoint():
    """测试原有的predict端点"""
    print("\n" + "=" * 60)
    print("测试predict端点...")
    print("=" * 60)
    
    try:
        # 准备测试数据
        test_data = {
            "img_src": "tests/test_images/image_custom.png",
            "model_name": "FasterNet"
        }
        
        print(f"发送请求: {test_data}")
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("响应结果:")
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            return True
        else:
            print(f"请求失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"predict端点测试失败: {str(e)}")
        return False

def test_detect_and_eval_endpoint():
    """测试新的detect-and-eval端点"""
    print("\n" + "=" * 60)
    print("测试v1/detect-and-eval端点...")
    print("=" * 60)
    
    try:
        # 测试1: 使用默认参数
        print("1. 测试默认参数...")
        test_data1 = {
            "img_src": "tests/test_images/image_custom.png"
        }
        
        print(f"发送请求: {test_data1}")
        response1 = requests.post(f"{BASE_URL}/v1/detect-and-eval", json=test_data1)
        print(f"状态码: {response1.status_code}")
        
        if response1.status_code == 200:
            result1 = response1.json()
            print("响应结果:")
            print(f"  success: {result1.get('success')}")
            print(f"  stage: {result1.get('stage')}")
            print(f"  message: {result1.get('message')}")
            print(f"  detected: {result1.get('detected')}")
            print(f"  total_time_delta: {result1.get('total_time_delta', 0):.4f}秒")
            
            if result1.get('detection_result'):
                det_result = result1['detection_result']
                print(f"  检测结果: 检测到 {det_result.get('detection_count', 0)} 个对象")
            
            if result1.get('evaluation_result'):
                eval_result = result1['evaluation_result']
                print(f"  评估结果:")
                print(f"    蛋白质: {eval_result.get('protein', 0):.2f}%")
                print(f"    油脂: {eval_result.get('oil', 0):.2f}%")
        else:
            print(f"请求失败: {response1.text}")
            return False
        
        # 测试2: 使用自定义参数
        print("\n2. 测试自定义参数...")
        test_data2 = {
            "img_src": "tests/test_images/image_custom.png",
            "model_name": "ResNet",
            "conf_threshold": 0.3,
            "iou_threshold": 0.5
        }
        
        print(f"发送请求: {test_data2}")
        response2 = requests.post(f"{BASE_URL}/v1/detect-and-eval", json=test_data2)
        print(f"状态码: {response2.status_code}")
        
        if response2.status_code == 200:
            result2 = response2.json()
            print("响应结果:")
            print(f"  success: {result2.get('success')}")
            print(f"  stage: {result2.get('stage')}")
            print(f"  message: {result2.get('message')}")
            print(f"  detected: {result2.get('detected')}")
            print(f"  model_name: {result2.get('model_name')}")
            print(f"  total_time_delta: {result2.get('total_time_delta', 0):.4f}秒")
        else:
            print(f"请求失败: {response2.text}")
            return False
        
        return True
        
    except Exception as e:
        print(f"detect-and-eval端点测试失败: {str(e)}")
        return False

def test_invalid_requests():
    """测试无效请求"""
    print("\n" + "=" * 60)
    print("测试无效请求...")
    print("=" * 60)
    
    try:
        # 测试缺少必需参数
        print("1. 测试缺少img_src参数...")
        test_data = {"model_name": "FasterNet"}
        response = requests.post(f"{BASE_URL}/v1/detect-and-eval", json=test_data)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:200]}...")
        
        # 测试无效的模型名称
        print("\n2. 测试无效的模型名称...")
        test_data = {
            "img_src": "tests/test_images/image_custom.png",
            "model_name": "InvalidModel"
        }
        response = requests.post(f"{BASE_URL}/v1/detect-and-eval", json=test_data)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:200]}...")
        
        # 测试无效的图像路径
        print("\n3. 测试无效的图像路径...")
        test_data = {
            "img_src": "nonexistent_image.jpg",
            "model_name": "FasterNet"
        }
        response = requests.post(f"{BASE_URL}/v1/detect-and-eval", json=test_data)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"响应: success={result.get('success')}, error={result.get('error', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"无效请求测试失败: {str(e)}")
        return False

def check_server_running():
    """检查服务器是否运行"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """主测试函数"""
    print("开始FastAPI端点测试")
    print("=" * 80)
    
    # 检查服务器是否运行
    if not check_server_running():
        print("❌ 服务器未运行！")
        print("请先启动服务器:")
        print("  cd backend")
        print("  python main.py")
        print("或者:")
        print("  python run_project.py")
        return
    
    print("✅ 服务器正在运行")
    
    # 运行所有测试
    tests = [
        ("根端点", test_root_endpoint),
        ("predict端点", test_predict_endpoint),
        ("detect-and-eval端点", test_detect_and_eval_endpoint),
        ("无效请求", test_invalid_requests)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n正在运行: {test_name}")
            results[test_name] = test_func()
            time.sleep(1)  # 给服务器一点休息时间
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
        print("🎉 所有API测试通过！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main()
