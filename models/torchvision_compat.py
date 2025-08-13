"""
Torchvision兼容性工具模块
处理不同版本torchvision之间的API差异
"""

import torchvision
import warnings
from typing import Optional, Any

def get_torchvision_version():
    """获取torchvision版本信息"""
    try:
        version = torchvision.__version__
        major, minor, patch = map(int, version.split('.')[:3])
        return major, minor, patch
    except:
        # 如果无法解析版本，假设是较新版本
        return 0, 13, 0

def is_new_torchvision():
    """
    检查是否为新版本的torchvision (>=0.13.0)
    新版本使用weights参数替代pretrained参数
    """
    major, minor, _ = get_torchvision_version()
    return major > 0 or (major == 0 and minor >= 13)

def create_model_with_compat(model_func, use_pretrained=False, **kwargs):
    """
    使用兼容性参数创建torchvision模型
    
    Args:
        model_func: torchvision模型函数 (如 resnet18, efficientnet_b0等)
        use_pretrained: 是否使用预训练权重
        **kwargs: 其他模型参数
        
    Returns:
        创建的模型实例
    """
    try:
        if is_new_torchvision():
            # 新版本使用weights参数
            if use_pretrained:
                # 对于新版本，使用默认权重
                return model_func(weights='DEFAULT', **kwargs)
            else:
                return model_func(weights=None, **kwargs)
        else:
            # 旧版本使用pretrained参数
            return model_func(pretrained=use_pretrained, **kwargs)
    except TypeError as e:
        # 如果参数错误，尝试另一种方式
        if 'weights' in str(e):
            # 新版本参数失败，尝试旧版本
            try:
                return model_func(pretrained=use_pretrained, **kwargs)
            except:
                raise e
        elif 'pretrained' in str(e):
            # 旧版本参数失败，尝试新版本
            try:
                if use_pretrained:
                    return model_func(weights='DEFAULT', **kwargs)
                else:
                    return model_func(weights=None, **kwargs)
            except:
                raise e
        else:
            raise e

def suppress_torchvision_warnings():
    """抑制torchvision的弃用警告"""
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 常用模型的兼容性包装函数
def resnet18_compat(use_pretrained=False, **kwargs):
    """ResNet18兼容性包装"""
    from torchvision.models import resnet18
    return create_model_with_compat(resnet18, use_pretrained, **kwargs)

def resnet50_compat(use_pretrained=False, **kwargs):
    """ResNet50兼容性包装"""
    from torchvision.models import resnet50
    return create_model_with_compat(resnet50, use_pretrained, **kwargs)

def efficientnet_b0_compat(use_pretrained=False, **kwargs):
    """EfficientNet-B0兼容性包装"""
    try:
        from torchvision.models import efficientnet_b0
        return create_model_with_compat(efficientnet_b0, use_pretrained, **kwargs)
    except ImportError:
        raise ImportError("EfficientNet not available in this torchvision version")

def vit_b_16_compat(use_pretrained=False, **kwargs):
    """Vision Transformer兼容性包装"""
    try:
        from torchvision.models import vit_b_16
        return create_model_with_compat(vit_b_16, use_pretrained, **kwargs)
    except ImportError:
        raise ImportError("Vision Transformer not available in this torchvision version")

# 版本信息打印
def print_torchvision_info():
    """打印torchvision版本和兼容性信息"""
    major, minor, patch = get_torchvision_version()
    is_new = is_new_torchvision()
    
    print(f"Torchvision版本: {major}.{minor}.{patch}")
    print(f"使用新API: {'是' if is_new else '否'}")
    print(f"推荐参数: {'weights=' if is_new else 'pretrained='}")

if __name__ == "__main__":
    # 测试兼容性功能
    print("Torchvision兼容性测试")
    print("=" * 40)
    
    print_torchvision_info()
    
    print("\n测试模型创建:")
    try:
        # 测试ResNet18
        model = resnet18_compat(use_pretrained=False)
        print("✓ ResNet18创建成功")
        
        # 测试EfficientNet
        try:
            model = efficientnet_b0_compat(use_pretrained=False)
            print("✓ EfficientNet-B0创建成功")
        except ImportError as e:
            print(f"⚠ EfficientNet-B0不可用: {e}")
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
