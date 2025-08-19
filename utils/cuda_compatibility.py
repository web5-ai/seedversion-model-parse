"""
CUDA兼容性检查和修复工具
解决PyTorch和torchvision版本不匹配导致的CUDA问题
"""

import os
import sys
import torch
import subprocess
from typing import Dict, Any, Optional
from utils.logging_config import get_logger

logger = get_logger("CUDA-Compatibility")

def check_cuda_environment() -> Dict[str, Any]:
    """
    全面检查CUDA环境
    
    Returns:
        Dict: 包含CUDA环境信息的字典
    """
    result = {
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "pytorch_version": None,
        "torchvision_version": None,
        "compatibility_issues": [],
        "recommendations": []
    }
    
    try:
        # 检查PyTorch CUDA支持
        result["cuda_available"] = torch.cuda.is_available()
        result["pytorch_version"] = torch.__version__
        
        if result["cuda_available"]:
            result["cuda_version"] = torch.version.cuda
            result["gpu_name"] = torch.cuda.get_device_name(0)
            
            # 检查torchvision版本
            try:
                import torchvision
                result["torchvision_version"] = torchvision.__version__
            except ImportError:
                result["compatibility_issues"].append("torchvision未安装")
                result["recommendations"].append("安装torchvision: pip install torchvision")
                
        # 检查版本兼容性
        _check_version_compatibility(result)
        
    except Exception as e:
        logger.error(f"检查CUDA环境时出错: {str(e)}")
        result["compatibility_issues"].append(f"环境检查失败: {str(e)}")
    
    return result

def _check_version_compatibility(result: Dict[str, Any]) -> None:
    """检查PyTorch和torchvision版本兼容性"""
    pytorch_version = result.get("pytorch_version", "")
    torchvision_version = result.get("torchvision_version", "")
    
    if not pytorch_version or not torchvision_version:
        return
    
    # 提取主版本号
    try:
        pytorch_major = pytorch_version.split('.')[0] + '.' + pytorch_version.split('.')[1]
        torchvision_major = torchvision_version.split('.')[0] + '.' + torchvision_version.split('.')[1]
        
        # 已知的兼容版本映射（基于官方兼容性矩阵）
        compatibility_map = {
            "2.5": "0.20",
            "2.4": "0.19",
            "2.3": "0.18",
            "2.2": "0.17",
            "2.1": "0.16",
            "2.0": "0.15",
            "1.13": "0.14",
            "1.12": "0.13",
            "1.11": "0.12",
            "1.10": "0.11"
        }
        
        expected_torchvision = compatibility_map.get(pytorch_major)
        if expected_torchvision and not torchvision_version.startswith(expected_torchvision):
            result["compatibility_issues"].append(
                f"版本不匹配: PyTorch {pytorch_version} 应配合 torchvision {expected_torchvision}.x"
            )
            result["recommendations"].append(
                f"重新安装兼容版本: pip install torch=={pytorch_version} torchvision=={expected_torchvision}.*"
            )
            
    except Exception as e:
        logger.warning(f"版本兼容性检查失败: {str(e)}")

def test_cuda_operations() -> Dict[str, Any]:
    """
    测试各种CUDA操作
    
    Returns:
        Dict: 测试结果
    """
    test_results = {
        "basic_cuda": False,
        "tensor_operations": False,
        "torchvision_nms": False,
        "memory_allocation": False,
        "errors": []
    }
    
    if not torch.cuda.is_available():
        test_results["errors"].append("CUDA不可用")
        return test_results
    
    try:
        # 测试基本CUDA操作
        test_tensor = torch.randn(100, 100).cuda()
        result = test_tensor.sum()
        result.cpu()
        test_results["basic_cuda"] = True
        logger.info("✓ 基本CUDA操作测试通过")
        
    except Exception as e:
        test_results["errors"].append(f"基本CUDA操作失败: {str(e)}")
        logger.error(f"✗ 基本CUDA操作失败: {str(e)}")
        return test_results
    
    try:
        # 测试张量操作
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        c = torch.matmul(a, b)
        c.cpu()
        test_results["tensor_operations"] = True
        logger.info("✓ 张量操作测试通过")
        
    except Exception as e:
        test_results["errors"].append(f"张量操作失败: {str(e)}")
        logger.error(f"✗ 张量操作失败: {str(e)}")
    
    try:
        # 测试torchvision NMS操作
        import torchvision.ops
        boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32).cuda()
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
        indices = torchvision.ops.nms(boxes, scores, 0.5)
        test_results["torchvision_nms"] = True
        logger.info("✓ torchvision NMS操作测试通过")
        
    except Exception as e:
        test_results["errors"].append(f"torchvision NMS失败: {str(e)}")
        logger.error(f"✗ torchvision NMS失败: {str(e)}")
    
    try:
        # 测试内存分配
        large_tensor = torch.randn(5000, 5000).cuda()
        del large_tensor
        torch.cuda.empty_cache()
        test_results["memory_allocation"] = True
        logger.info("✓ 内存分配测试通过")
        
    except Exception as e:
        test_results["errors"].append(f"内存分配失败: {str(e)}")
        logger.error(f"✗ 内存分配失败: {str(e)}")
    
    return test_results

def suggest_fixes(cuda_env: Dict[str, Any], test_results: Dict[str, Any]) -> list:
    """
    根据检查结果提供修复建议
    
    Args:
        cuda_env: CUDA环境信息
        test_results: 测试结果
        
    Returns:
        list: 修复建议列表
    """
    suggestions = []
    
    if not cuda_env["cuda_available"]:
        suggestions.extend([
            "1. 检查NVIDIA驱动是否正确安装",
            "2. 确认GPU硬件正常工作",
            "3. 重新安装支持CUDA的PyTorch版本"
        ])
        return suggestions
    
    if cuda_env["compatibility_issues"]:
        suggestions.extend(cuda_env["recommendations"])
    
    if not test_results["torchvision_nms"]:
        suggestions.extend([
            "torchvision NMS操作失败的解决方案:",
            "1. 重新安装匹配的PyTorch和torchvision版本",
            "2. 使用conda安装以确保版本兼容性:",
            "   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia",
            "3. 或者使用pip安装特定版本:",
            f"   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        ])
    
    if not test_results["memory_allocation"]:
        suggestions.extend([
            "内存分配问题的解决方案:",
            "1. 检查GPU内存是否充足",
            "2. 关闭其他占用GPU的程序",
            "3. 降低批处理大小"
        ])
    
    return suggestions

def run_compatibility_check() -> Dict[str, Any]:
    """
    运行完整的兼容性检查
    
    Returns:
        Dict: 完整的检查结果
    """
    logger.info("开始CUDA兼容性检查...")
    
    # 检查环境
    cuda_env = check_cuda_environment()
    
    # 运行测试
    test_results = test_cuda_operations()
    
    # 生成建议
    suggestions = suggest_fixes(cuda_env, test_results)
    
    # 汇总结果
    result = {
        "environment": cuda_env,
        "tests": test_results,
        "suggestions": suggestions,
        "overall_status": "pass" if test_results.get("basic_cuda", False) else "fail"
    }
    
    # 输出总结
    logger.info("=" * 60)
    logger.info("CUDA兼容性检查结果")
    logger.info("=" * 60)
    logger.info(f"CUDA可用: {cuda_env['cuda_available']}")
    if cuda_env['cuda_available']:
        logger.info(f"GPU: {cuda_env['gpu_name']}")
        logger.info(f"CUDA版本: {cuda_env['cuda_version']}")
        logger.info(f"PyTorch版本: {cuda_env['pytorch_version']}")
        logger.info(f"torchvision版本: {cuda_env['torchvision_version']}")
    
    logger.info(f"基本CUDA操作: {'✓' if test_results['basic_cuda'] else '✗'}")
    logger.info(f"张量操作: {'✓' if test_results['tensor_operations'] else '✗'}")
    logger.info(f"torchvision NMS: {'✓' if test_results['torchvision_nms'] else '✗'}")
    logger.info(f"内存分配: {'✓' if test_results['memory_allocation'] else '✗'}")
    
    if suggestions:
        logger.info("\n修复建议:")
        for suggestion in suggestions:
            logger.info(f"  {suggestion}")
    
    logger.info("=" * 60)
    
    return result

if __name__ == "__main__":
    run_compatibility_check()
