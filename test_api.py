import requests
import json

def test_basic_api():
    """测试基础API"""
    print("=== 测试基础API (/predict) ===")
    url = "http://localhost:8123/predict"
    data = {
        "image_url": "https://img.remit.ee/api/file/BQACAgUAAyEGAASHRsPbAAKX2GikNkRtBNIpPGN6Vt984gSt1NfPAALAGgACDkAoVaHTlLruxLFJNgQ.jpg",
        "model": "ResNet"
    }

    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_v1_api():
    """测试v1 API"""
    print("\n=== 测试v1 API (/v1/predict) ===")
    url = "http://localhost:8123/v1/predict"
    data = {
        "image_url": "https://img.remit.ee/api/file/BQACAgUAAyEGAASHRsPbAAKX2GikNkRtBNIpPGN6Vt984gSt1NfPAALAGgACDkAoVaHTlLruxLFJNgQ.jpg",
        "model": "ResNet"
    }

    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_v1_api_with_thresholds():
    """测试v1 API（带阈值参数）"""
    print("\n=== 测试v1 API 带阈值参数 ===")
    url = "http://localhost:8123/v1/predict"
    data = {
        "image_url": "https://img.remit.ee/api/file/BQACAgUAAyEGAASHRsPbAAKX2GikNkRtBNIpPGN6Vt984gSt1NfPAALAGgACDkAoVaHTlLruxLFJNgQ.jpg",
        "model": "ResNet",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45
    }

    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_basic_api()
    test_v1_api()
    test_v1_api_with_thresholds()
