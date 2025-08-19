#!/usr/bin/env python3
"""
API测试脚本 - 验证API文档的准确性
测试v1和v2 API接口，确保响应格式与文档一致
"""

import requests
import json
import time

# API配置
BASE_URL = "http://localhost:8123"
TEST_IMAGE_URL = "https://img.remit.ee/api/file/BQACAgUAAyEGAASHRsPbAAKX2GikNkRtBNIpPGN6Vt984gSt1NfPAALAGgACDkAoVaHTlLruxLFJNgQ.jpg"

def test_v1_api():
    """测试v1 API接口"""
    print("=" * 50)
    print("测试 V1 API (/v1/predict)")
    print("=" * 50)
    
    url = f"{BASE_URL}/v1/predict"
    data = {
        "image_url": TEST_IMAGE_URL,
        "model": "ResNet",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45
    }
    
    try:
        print(f"请求URL: {url}")
        print(f"请求数据: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, json=data, timeout=30)
        result = response.json()
        
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应数据: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # 验证响应格式
        if response.status_code == 200:
            expected_fields = ["detected", "protein", "oil", "message", "time_delta"]
            missing_fields = [field for field in expected_fields if field not in result]

            if missing_fields:
                print(f"❌ 缺少字段: {missing_fields}")
            else:
                print("✅ V1 API响应格式正确")
                
            # 检查是否包含seed_info（应该不包含）
            if "seed_info" in str(result):
                print("❌ 响应中仍包含seed_info字段")
            else:
                print("✅ 已成功移除seed_info字段")
        else:
            print(f"❌ API调用失败: {result}")
            
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")

def test_v2_api():
    """测试v2 API接口"""
    print("\n" + "=" * 50)
    print("测试 V2 API (/v2/predict)")
    print("=" * 50)
    
    url = f"{BASE_URL}/v2/predict"
    data = {
        "image_url": TEST_IMAGE_URL,
        "model": "ResNet",
        "conf_threshold": 0.3,
        "iou_threshold": 0.4
    }
    
    try:
        print(f"请求URL: {url}")
        print(f"请求数据: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, json=data, timeout=30)
        result = response.json()
        
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应数据: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # 验证响应格式
        if response.status_code == 200:
            expected_fields = ["success", "stage", "message", "detected", "detection_result", "total_time_delta"]
            missing_fields = [field for field in expected_fields if field not in result]
            
            if missing_fields:
                print(f"❌ 缺少字段: {missing_fields}")
            else:
                print("✅ V2 API响应格式正确")
                
            # 检查是否包含seed_info（应该不包含）
            if "seed_info" in str(result):
                print("❌ 响应中仍包含seed_info字段")
            else:
                print("✅ 已成功移除seed_info字段")
        else:
            print(f"❌ API调用失败: {result}")
            
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")

def test_parameter_effects():
    """测试不同参数的效果"""
    print("\n" + "=" * 50)
    print("测试参数效果对比")
    print("=" * 50)
    
    test_cases = [
        {"name": "默认参数", "conf": 0.25, "iou": 0.45},
        {"name": "严格检测", "conf": 0.4, "iou": 0.45},
        {"name": "宽松检测", "conf": 0.15, "iou": 0.45},
    ]
    
    url = f"{BASE_URL}/v1/predict"
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        data = {
            "image_url": TEST_IMAGE_URL,
            "model": "ResNet",
            "conf_threshold": case["conf"],
            "iou_threshold": case["iou"]
        }
        
        try:
            response = requests.post(url, json=data, timeout=30)
            result = response.json()
            
            if response.status_code == 200 and result.get("detected"):
                protein = result.get("protein", 0)
                oil = result.get("oil", 0)
                time_delta = result.get("time_delta", 0)
                message = result.get("message", "")

                print(f"状态: {message}")
                print(f"蛋白质: {protein:.2f}%")
                print(f"油脂: {oil:.2f}%")
                print(f"耗时: {time_delta:.3f}秒")
            else:
                print(f"❌ 测试失败: {result}")
                
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
        
        time.sleep(1)  # 避免请求过快

def main():
    """主函数"""
    print("🚀 开始API测试")
    print(f"测试服务器: {BASE_URL}")
    print(f"测试图像: {TEST_IMAGE_URL}")
    
    # 测试v1 API
    test_v1_api()
    
    # 等待一秒避免请求过快
    time.sleep(1)
    
    # 测试v2 API
    test_v2_api()
    
    # 等待一秒避免请求过快
    time.sleep(1)
    
    # 测试参数效果
    test_parameter_effects()
    
    print("\n" + "=" * 50)
    print("🎉 API测试完成！")
    print("=" * 50)
    print("\n📖 查看完整API文档: docs/API_Documentation.md")

if __name__ == "__main__":
    main()
