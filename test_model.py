import os
import sys

# 添加当前目录到Python路径，确保可以导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
import argparse
import random
import datetime
from utils.model_loader import ModelLoader
from config import MODEL_CONFIG, IMAGE_CONFIG, OUTPUT_CONFIG, SYSTEM_CONFIG
from utils.environment import check_dependencies, setup_environment

# 设置日志
def setup_logger(name="ModelTest", level=None):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
    
    Returns:
        配置好的日志记录器
    """
    if level is None:
        level = SYSTEM_CONFIG["log_level"]
    
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    logging.basicConfig(
        level=level_map.get(level, logging.INFO),
        format=SYSTEM_CONFIG["log_format"],
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(name)

logger = setup_logger()

# 设置随机种子，确保结果可重复
def set_seed(seed=None):
    """
    设置随机种子以确保结果可重复
    
    Args:
        seed: 随机种子
    """
    if seed is None:
        seed = SYSTEM_CONFIG["default_seed"]
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # 固定卷积算法以提高性能
    torch.backends.cudnn.benchmark = False # 关闭动态卷积算法
    logger.info(f"已设置随机种子: {seed}")

def load_model(model_name = 'FasterNet',model_path = None, device='cpu'):
    """
    加载预训练模型
    
    Args:
        model_path: 模型路径
        device: 运行设备
    
    Returns:
        加载好的模型
    """
    try:
        # 使用ModelLoader加载模型
        model_loader = ModelLoader(model_path, debug=True)
        model_loader._load_model(model_name= model_name)
        model = model_loader.model
        model.to(device)
        model.eval()
        logger.info(f"模型成功加载: {model_path}")
        return model, model_loader
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return None, None

def preprocess_image(image_path, model_loader=None):
    """
    预处理图像
    
    Args:
        image_path: 图像路径
        model_loader: 模型加载器，用于获取预处理方法
    
    Returns:
        预处理后的图像张量
    """
    try:
        image = Image.open(image_path).convert('RGB')
        
        if model_loader:
            # 使用模型加载器中的预处理方法
            tensor = model_loader.preprocess_image(image)
        else:
            # 使用已有的ImageProcessor
            from utils.image_processor import ImageProcessor
            processor = ImageProcessor()
            tensor = processor.preprocess_image(image_path)
        
        return tensor
    except Exception as e:
        logger.error(f"图像预处理失败: {str(e)}")
        return None

def predict(model, image_tensor, device='cpu'):
    """
    使用模型进行预测
    
    Args:
        model: 加载的模型
        image_tensor: 预处理后的图像张量
        device: 运行设备
    
    Returns:
        预测结果
    """
    try:
        with torch.no_grad(): # 禁用梯度计算以节省内存和计算
            image_tensor = image_tensor.to(device) # 将图像移动到指定设备
            output = model(image_tensor)
            return output
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        return None

def generate_text_report(output_np, components, save_path=None, model_name=None):
    """
    生成文本报告
    
    Args:
        output_np: 模型输出的numpy数组
        components: 成分名称列表
        save_path: 保存路径
        model_name: 模型文件名
    """
    if len(output_np) !=2:
        logger.error(f"输出维度错误，应为2，实际为{len(output_np)}")
        return
    # 打印数值结果
    logger.info("=== 油菜籽成分含量预测报告 ===")
    logger.info("成分含量预测结果:")
    for i, comp in enumerate(components):
        logger.info(f"{comp}: {output_np[i]:.4f}")
    
    # 添加详细的文字结论
    logger.info("预测结论:")
    logger.info(f'预测模型: {model_name if model_name else "未知模型"}')
    
    logger.info("=== 报告结束 ===")
    
    return {
        'protein': f'{output_np[0]:.4f}',
        'oil': f'{output_np[1]:.4f}',
    }

def save_chart(output_np, components, save_path):
    """
    保存图表
    
    Args:
        output_np: 模型输出的numpy数组
        components: 成分名称列表
        save_path: 保存路径
    """
    if not save_path:
        logger.info("跳过图表生成，仅输出文本报告")
        return
    
    # 使用英文标签避免中文字体问题
    component_labels = ["Oleic", "Linoleic", "Linolenic", "Palmitic", "Stearic"]
    
    plt.figure(figsize=OUTPUT_CONFIG["chart_size"])
    plt.bar(component_labels, output_np)
    plt.title("Rapeseed Component Prediction")
    plt.xlabel('Components')
    plt.ylabel('Predicted Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # 关闭图表而不显示
    logger.info(f"图表已保存至: {save_path}")

def visualize_results(output, class_names=None, save_path=None, model_name = None):
    """
    可视化预测结果
    
    Args:
        output: 模型输出
        class_names: 类别名称列表（如果是分类任务）
        save_path: 保存路径
        model_name: 模型文件名
    """
    # 将输出转换为numpy数组
    output_np = output.cpu().numpy().flatten()
    
    # 如果输出维度大于预期的成分数量，只取前几个值
    expected_components = MODEL_CONFIG["expected_components"]
    if len(output_np) > expected_components:
        logger.warning(f"模型输出维度({len(output_np)})大于预期成分数量({expected_components})，只取前{expected_components}个值")
        output_np = output_np[:expected_components]
    
    components = MODEL_CONFIG["component_names"][:len(output_np)] if class_names is None else class_names[:len(output_np)]
    
    # 只有在需要图表时才保存图表
    if save_path:
        save_chart(output_np, components, save_path)
    else:
        logger.info("仅生成文本报告，不生成图表")
    
    # 生成文本报告
    return generate_text_report(output_np, components, save_path, model_name=model_name)

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='测试油菜籽成分预测模型')
    parser.add_argument('--model', type=str, default=MODEL_CONFIG["model_path"], help='模型路径')
    parser.add_argument('--image', type=str, default=IMAGE_CONFIG["default_image_path"], help='测试图像路径')
    parser.add_argument('--output', type=str, default=OUTPUT_CONFIG["default_output_path"], help='结果保存路径')
    parser.add_argument('--seed', type=int, default=SYSTEM_CONFIG["default_seed"], help='随机种子')
    parser.add_argument('--no-chart', action='store_true', help='不生成图表，只输出文本报告')
    parser.add_argument('--check-env', action='store_true', help='检查环境并安装依赖')
    return parser.parse_args()

def auto_model_test(except_models= [], test_image=None):
    '''
    从main()复制过来改了一下
    使用自动化参数测试weights下的所有模型
    '''

    # 读取weights文件夹下的所有模型
    model_paths = [os.path.join('weights', f) for f in os.listdir('weights') if f.endswith('.pt')]
    logger.info("发现{}个模型文件：\n{}".format(len(model_paths), '\n'.join(model_paths))) # 这里输出模型路径列表

    # model_paths = [
    #     'weights/fasternet_model.pt',
    #     'weights/ResNet18_base.pt',
    #     'weights/swin.pt',
    #     'weights/vanillanet.pt',
    #     'weights/mpvit.pt',
    #     'weights/efficientnet.pt'
    # ]
    res = []
    for model_path in model_paths:
        if os.path.basename(model_path).split('.')[0] in except_models: # 跳过except_models中的模型
            logger.info(f"跳过模型: {model_path}")
            continue
        r:dict = test_model(model_path, test_image=test_image)
        res.append(r)
    with open ('output.txt', 'w') as f: # 输出结果到文件
        for r in res: # 输出结果
            for key, value in r.items():
                f.write(f"{key}: {value}\n")


def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查环境
    if args.check_env:
        logger.info("正在检查环境...")
        if not check_dependencies() or not setup_environment():
            logger.error("环境检查失败，请解决上述问题后重试")
            return
        logger.info("环境检查通过!")
        return
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置模型路径和测试图像路径
    model_path = args.model
    # model_path =  'weights/ResNet18_best.pt' #args.model
    test_image_path = args.image
    
    # 默认不生成图表，只输出文本报告
    output_path = None
    args.no_chart = True  # 强制设置为不生成图表
    
    # 输出运行模式信息
    logger.info("运行模式: 仅文本报告，不生成图表")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"错误: 模型文件不存在: {model_path}")
        logger.info(f"请确保模型文件位于正确位置，或使用 --model 参数指定正确的路径")
        return
    
    if not os.path.exists(test_image_path):
        logger.error(f"错误: 测试图像不存在: {test_image_path}")
        logger.info(f"请确保测试图像位于正确位置，或使用 --image 参数指定正确的路径")
        return
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model, model_loader = load_model('ResNet', model_path, device)
    
    if model is None:
        return

    # # 打印模型结构
    # logger.info("\n模型结构:")
    # logger.info(str(model))
    
    # 预处理图像
    image_tensor = preprocess_image(test_image_path, model_loader)
    
    if image_tensor is None:
        return
    
    # 进行预测
    output = predict(model, image_tensor, device)
    
    if output is None:
        return
    
    # 显示预测结果
    logger.info("\n预测结果:")
    logger.info(f"输出张量形状: {output.shape}")
    logger.info(f"输出值: {output.cpu().numpy()}")
    
    # 可视化结果
    component_names = MODEL_CONFIG["component_names"][:output.shape[1]]
    visualize_results(output, component_names, output_path)
    
    logger.info("\n测试完成!")

def test_model(model_path = 'weights/ResNet18_best.pt', test_image = None):
    args = parse_arguments()

    # 设置随机种子
    set_seed(args.seed)

    test_image_path = args.image
    
    # 默认不生成图表和文本文件，只在终端输出文本报告
    args.no_chart = True  # 强制设置为不生成图表

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model_loader = ModelLoader(model_path, debug=True)
    model_name = model_path.split('\\')[-1].split('.')[0] # 从模型路径中提取模型名称
    model_loader._load_model(model_name= model_name) # 这里可以根据模型文件名称选择模型结构，这里默认选择ResNet
    if model_loader.model.state_dict() is None:
        logger.error(f"模型加载失败: {model_path}")
        return
    model_loader._analyze_model()
    model_loader._analyze_state_dict()
    if test_image is None:
        logger.info(f"不进行预测测试")
    else:
        img = Image.open(test_image)
        img_tensor = model_loader.preprocess_image(img, 224) if model_loader.model_name != 'Swin' else model_loader.preprocess_image(img,256)
        output = model_loader.predict(img_tensor)
        logger.info(f'模型{model_path} 预测结果: 蛋白质:{output[1]}, 油脂:{output[0]}')
        return {
            'model': model_name,
           'oil': f'{output[0]:.4f}',
           'protein': f'{output[1]:.4f}',
        }

if __name__ == "__main__":
    # test_model('weights\VanillaNet.pt')
    # test_model('weights\FasterNet.pt')
    exceptmodel = []
    img = r'tests\test_images\image_custom.png'
    auto_model_test(except_models=exceptmodel, test_image= img)
    # main()