"""
环境测试脚本，用于测试当前环境下的各种变量情况
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 添加当前目录到Python路径，确保可以导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils.model_loader import ModelLoader
from config import MODEL_CONFIG, IMAGE_CONFIG, SYSTEM_CONFIG
from utils.environment import check_dependencies, setup_environment

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='环境测试工具')
    parser.add_argument('--model', type=str, default="ResNet", help='模型名称，默认为ResNet')
    parser.add_argument('--image', type=str, default=IMAGE_CONFIG["default_image_path"], help='测试图像路径')
    parser.add_argument('--seed', type=int, default=SYSTEM_CONFIG["default_seed"], help='随机种子')
    parser.add_argument('--output', type=str, default=None, help='结果保存路径，默认为当前目录下的env_test_结果.json')
    parser.add_argument('--device', type=str, default=MODEL_CONFIG["device"], help='设备，默认为config中的设置')
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查环境
    print("正在检查环境...")
    if not check_dependencies() or not setup_environment():
        print("环境检查失败，请解决上述问题后重试")
        return
    print("环境检查通过!")
    
    # 初始化模型加载器
    print(f"初始化模型加载器，设备: {args.device}")
    model_loader = ModelLoader(debug=True, device=args.device)
    
    # 运行环境测试
    print(f"开始运行环境测试: 模型={args.model}, 图像={args.image}, 种子={args.seed}")
    result = model_loader.env_test(args.model, args.image, args.seed)
    
    # 保存结果
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        args.output = f"env_test_{timestamp}.json"
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"环境测试完成，结果已保存至: {args.output}")
    
    # 打印关键信息
    print("\n=== 环境测试结果摘要 ===")
    print(f"Python版本: {result['environment'].get('python_version', '未知')}")
    print(f"PyTorch版本: {result['environment'].get('torch_version', '未知')}")
    print(f"CUDA可用: {result['environment'].get('cuda_available', '未知')}")
    if result['environment'].get('cuda_available'):
        print(f"CUDA版本: {result['environment'].get('cuda_version', '未知')}")
        print(f"GPU名称: {result['environment'].get('gpu_name', '未知')}")
    
    print(f"\n模型: {result['model'].get('name', '未知')}")
    print(f"设备: {result['model'].get('device', '未知')}")
    print(f"参数数量: {result['model'].get('total_params', '未知')}")
    
    print(f"\n图像路径: {result['image'].get('original_path', '未知')}")
    print(f"原始图像哈希: {result['image'].get('original_hash', '未知')}")
    print(f"处理后图像哈希: {result['image'].get('processed_hash', '未知')}")
    print(f"哈希是否变化: {result['image'].get('hash_changed', '未知')}")
    
    if 'protein' in result['prediction'] and 'oil' in result['prediction']:
        print(f"\n预测结果:")
        print(f"蛋白质: {result['prediction']['protein']}")
        print(f"油脂: {result['prediction']['oil']}")
    
    print(f"\n预测结果哈希: {result['prediction'].get('hash', '未知')}")
    print(f"随机种子: {result['seed_info'].get('set_seed', '未知')}")
    print(f"PyTorch确定性: {result['seed_info'].get('torch_deterministic', '未知')}")
    print(f"PyTorch基准测试: {result['seed_info'].get('torch_benchmark', '未知')}")
    print("=== 摘要结束 ===\n")
    
    print(f"完整结果请查看: {args.output}")

if __name__ == "__main__":
    main()
