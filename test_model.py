import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
import argparse
import random
from models.model_loader import ModelLoader

# 设置随机种子，确保结果可重复
def set_seed(seed=42):
    """
    设置随机种子以确保结果可重复
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"已设置随机种子: {seed}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ModelTest")

def load_model(model_path, device='cpu'):
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
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            return output
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        return None

def visualize_results(output, class_names=None, save_path=None):
    """
    可视化预测结果
    
    Args:
        output: 模型输出
        class_names: 类别名称列表（如果是分类任务）
        save_path: 保存路径
    """
    # 将输出转换为numpy数组
    output_np = output.cpu().numpy().flatten()
    
    # 如果输出维度大于预期的成分数量，只取前几个值
    expected_components = 5  # 预期的成分数量
    if len(output_np) > expected_components:
        logger.warning(f"模型输出维度({len(output_np)})大于预期成分数量({expected_components})，只取前{expected_components}个值")
        output_np = output_np[:expected_components]
    
    components = [f"成分{i+1}" for i in range(len(output_np))] if class_names is None else class_names[:len(output_np)]
    
    # 如果需要保存图表，仍然生成但不显示
    if save_path:
        plt.figure(figsize=(10, 6))
        plt.bar(components, output_np)
        plt.title('油菜籽成分含量预测')
        plt.xlabel('成分')
        plt.ylabel('含量预测值')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()  # 关闭图表而不显示
        logger.info(f"图表已保存至: {save_path}")
    
    # 打印数值结果
    logger.info("\n=== 油菜籽成分含量预测报告 ===")
    logger.info("\n成分含量预测结果:")
    for i, comp in enumerate(components):
        logger.info(f"{comp}: {output_np[i]:.4f}")
    
    # 添加详细的文字结论
    logger.info("\n预测结论:")
    
    # 找出含量最高的成分
    max_index = np.argmax(output_np)
    max_component = components[max_index]
    max_value = output_np[max_index]
    
    # 找出含量最低的成分
    min_index = np.argmin(output_np)
    min_component = components[min_index]
    min_value = output_np[min_index]
    
    # 计算平均含量
    avg_value = np.mean(output_np)
    
    # 输出结论
    logger.info(f"1. 该油菜籽样本中含量最高的成分是 {max_component}，含量为 {max_value:.4f}")
    logger.info(f"2. 含量最低的成分是 {min_component}，含量为 {min_value:.4f}")
    logger.info(f"3. 所有成分的平均含量为 {avg_value:.4f}")
    
    # 根据含量高低对成分进行排序
    sorted_indices = np.argsort(output_np)[::-1]  # 从高到低排序
    logger.info("4. 各成分含量从高到低排序:")
    for i, idx in enumerate(sorted_indices):
        logger.info(f"   {i+1}. {components[idx]}: {output_np[idx]:.4f}")
    
    # 添加一些简单的品质评估（示例）
    logger.info("\n5. 品质评估:")
    if output_np[0] > 0.5:  # 假设油酸含量高于0.5为优质
        logger.info(f"   油酸含量较高 ({output_np[0]:.4f})，品质较好")
    else:
        logger.info(f"   油酸含量较低 ({output_np[0]:.4f})，品质一般")
    
    if output_np[1] > 0.3:  # 假设亚油酸含量高于0.3为优质
        logger.info(f"   亚油酸含量较高 ({output_np[1]:.4f})，营养价值较高")
    else:
        logger.info(f"   亚油酸含量较低 ({output_np[1]:.4f})，营养价值一般")
    
    # 将结果保存到文本文件
    if save_path:
        txt_path = os.path.splitext(save_path)[0] + "_report.txt"
        try:
            import datetime
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=== 油菜籽成分含量预测报告 ===\n\n")
                
                f.write("成分含量预测结果:\n")
                for i, comp in enumerate(components):
                    f.write(f"{comp}: {output_np[i]:.4f}\n")
                
                f.write("\n预测结论:\n")
                f.write(f"1. 该油菜籽样本中含量最高的成分是 {max_component}，含量为 {max_value:.4f}\n")
                f.write(f"2. 含量最低的成分是 {min_component}，含量为 {min_value:.4f}\n")
                f.write(f"3. 所有成分的平均含量为 {avg_value:.4f}\n")
                
                f.write("\n4. 各成分含量从高到低排序:\n")
                for i, idx in enumerate(sorted_indices):
                    f.write(f"   {i+1}. {components[idx]}: {output_np[idx]:.4f}\n")
                
                f.write("\n5. 品质评估:\n")
                if output_np[0] > 0.5:
                    f.write(f"   油酸含量较高 ({output_np[0]:.4f})，品质较好\n")
                else:
                    f.write(f"   油酸含量较低 ({output_np[0]:.4f})，品质一般\n")
                
                if output_np[1] > 0.3:
                    f.write(f"   亚油酸含量较高 ({output_np[1]:.4f})，营养价值较高\n")
                else:
                    f.write(f"   亚油酸含量较低 ({output_np[1]:.4f})，营养价值一般\n")
                
                f.write(f"\n报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            logger.info(f"\n文本报告已保存至: {txt_path}")
        except Exception as e:
            logger.error(f"保存报告失败: {str(e)}")
    
    logger.info("\n=== 报告结束 ===")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试油菜籽成分预测模型')
    parser.add_argument('--model', type=str, default='weights/fasternet_model.pt', help='模型路径')
    parser.add_argument('--image', type=str, default='tests/test_images/image_custom.png', help='测试图像路径')
    parser.add_argument('--output', type=str, default='results/prediction_result.png', help='结果保存路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置模型路径和测试图像路径
    model_path = args.model
    test_image_path = args.image
    output_path = args.output
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"错误: 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(test_image_path):
        logger.error(f"错误: 测试图像不存在: {test_image_path}")
        return
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model, model_loader = load_model(model_path, device)
    
    if model is None:
        return
    
    # 打印模型结构
    logger.info("\n模型结构:")
    logger.info(str(model))
    
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
    # 这里可以根据实际情况定义成分名称
    component_names = ["油酸", "亚油酸", "亚麻酸", "棕榈酸", "硬脂酸"][:output.shape[1]]
    visualize_results(output, component_names, output_path)
    
    logger.info("\n测试完成!")

if __name__ == "__main__":
    main()