"""
比较不同启动方式的环境测试结果
"""

import os
import sys
import json
import argparse
from datetime import datetime

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='比较环境测试结果')
    parser.add_argument('--file1', type=str, required=True, help='第一个环境测试结果文件路径')
    parser.add_argument('--file2', type=str, required=True, help='第二个环境测试结果文件路径')
    parser.add_argument('--output', type=str, default=None, help='比较结果保存路径')
    return parser.parse_args()

def load_json_file(file_path):
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的JSON数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件失败: {file_path}, 错误: {str(e)}")
        return None

def compare_dicts(dict1, dict2, path=""):
    """
    比较两个字典的差异
    
    Args:
        dict1: 第一个字典
        dict2: 第二个字典
        path: 当前路径
        
    Returns:
        差异列表
    """
    differences = []
    
    # 比较dict1中的键
    for key in dict1:
        new_path = f"{path}.{key}" if path else key
        
        # 如果key不在dict2中
        if key not in dict2:
            differences.append(f"键 '{new_path}' 在第二个文件中不存在")
            continue
        
        # 如果值类型不同
        if type(dict1[key]) != type(dict2[key]):
            differences.append(f"键 '{new_path}' 的类型不同: {type(dict1[key]).__name__} vs {type(dict2[key]).__name__}")
            continue
        
        # 如果值是字典，递归比较
        if isinstance(dict1[key], dict):
            differences.extend(compare_dicts(dict1[key], dict2[key], new_path))
        # 如果值是列表，比较列表
        elif isinstance(dict1[key], list):
            # 简单比较列表长度
            if len(dict1[key]) != len(dict2[key]):
                differences.append(f"键 '{new_path}' 的列表长度不同: {len(dict1[key])} vs {len(dict2[key])}")
            # 对于简单列表，比较内容
            elif all(not isinstance(item, (dict, list)) for item in dict1[key] + dict2[key]):
                if dict1[key] != dict2[key]:
                    differences.append(f"键 '{new_path}' 的列表内容不同")
        # 对于其他类型，直接比较值
        elif dict1[key] != dict2[key]:
            # 对于数值型，计算差异百分比
            if isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
                if dict1[key] == 0 and dict2[key] == 0:
                    # 两个值都为0，没有差异
                    pass
                elif dict1[key] == 0 or dict2[key] == 0:
                    # 一个值为0，无法计算百分比
                    differences.append(f"键 '{new_path}' 的值不同: {dict1[key]} vs {dict2[key]}")
                else:
                    # 计算差异百分比
                    diff_percent = abs(dict1[key] - dict2[key]) / max(abs(dict1[key]), abs(dict2[key])) * 100
                    differences.append(f"键 '{new_path}' 的值不同: {dict1[key]} vs {dict2[key]} (差异: {diff_percent:.2f}%)")
            else:
                differences.append(f"键 '{new_path}' 的值不同: {dict1[key]} vs {dict2[key]}")
    
    # 检查dict2中有但dict1中没有的键
    for key in dict2:
        new_path = f"{path}.{key}" if path else key
        if key not in dict1:
            differences.append(f"键 '{new_path}' 在第一个文件中不存在")
    
    return differences

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载环境测试结果
    data1 = load_json_file(args.file1)
    data2 = load_json_file(args.file2)
    
    if data1 is None or data2 is None:
        print("加载文件失败，无法比较")
        return
    
    # 比较差异
    differences = compare_dicts(data1, data2)
    
    # 输出差异
    print(f"找到 {len(differences)} 处差异:")
    for diff in differences:
        print(f"- {diff}")
    
    # 重点关注预测结果和哈希值
    print("\n=== 重点关注项 ===")
    
    # 比较预测结果
    if "prediction" in data1 and "prediction" in data2:
        print("\n预测结果比较:")
        if "raw_output" in data1["prediction"] and "raw_output" in data2["prediction"]:
            output1 = data1["prediction"]["raw_output"]
            output2 = data2["prediction"]["raw_output"]
            if len(output1) == len(output2):
                for i, (val1, val2) in enumerate(zip(output1, output2)):
                    diff = abs(val1 - val2) if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else "N/A"
                    print(f"输出[{i}]: {val1} vs {val2}, 差异: {diff}")
            else:
                print(f"输出长度不同: {len(output1)} vs {len(output2)}")
        
        # 比较哈希值
        if "hash" in data1["prediction"] and "hash" in data2["prediction"]:
            hash1 = data1["prediction"]["hash"]
            hash2 = data2["prediction"]["hash"]
            print(f"\n哈希值比较: {'相同' if hash1 == hash2 else '不同'}")
            print(f"哈希值1: {hash1}")
            print(f"哈希值2: {hash2}")
    
    # 比较环境变量
    if "environment" in data1 and "environment" in data2:
        if "env_vars" in data1["environment"] and "env_vars" in data2["environment"]:
            print("\n环境变量比较:")
            env_vars1 = data1["environment"]["env_vars"]
            env_vars2 = data2["environment"]["env_vars"]
            
            for key in set(env_vars1.keys()) | set(env_vars2.keys()):
                val1 = env_vars1.get(key, "不存在")
                val2 = env_vars2.get(key, "不存在")
                if val1 != val2:
                    print(f"{key}: {val1} vs {val2}")
    
    # 比较进程信息
    if "environment" in data1 and "environment" in data2:
        print("\n进程信息比较:")
        for key in ["process_id", "parent_process_id", "process_name", "process_cmdline"]:
            if key in data1["environment"] and key in data2["environment"]:
                val1 = data1["environment"][key]
                val2 = data2["environment"][key]
                print(f"{key}: {val1} vs {val2}")
    
    # 比较PyTorch和CUDA信息
    if "environment" in data1 and "environment" in data2:
        print("\nPyTorch和CUDA信息比较:")
        for key in ["torch_threads", "cudnn_deterministic", "cudnn_benchmark"]:
            if key in data1["environment"] and key in data2["environment"]:
                val1 = data1["environment"][key]
                val2 = data2["environment"][key]
                print(f"{key}: {val1} vs {val2}")
    
    # 保存比较结果
    if args.output:
        result = {
            "file1": args.file1,
            "file2": args.file2,
            "differences_count": len(differences),
            "differences": differences,
            "comparison_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print(f"\n比较结果已保存至: {args.output}")

if __name__ == "__main__":
    main()
