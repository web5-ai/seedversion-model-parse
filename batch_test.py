import os
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse
from test_model import load_model, preprocess_image, predict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BatchTest")

def batch_test(model_path, image_folder, output_csv=None, device='cpu'):
    """
    批量测试多张图像
    
    Args:
        model_path: 模型路径
        image_folder: 图像文件夹路径
        output_csv: 输出CSV文件路径
        device: 运行设备
    """
    # 加载模型
    model, model_loader = load_model(model_path, device)
    if model is None:
        return
    
    # 获取所有图像文件
    image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                 glob.glob(os.path.join(image_folder, "*.jpeg")) + \
                 glob.glob(os.path.join(image_folder, "*.png"))
    
    if not image_files:
        logger.error(f"错误: 在 {image_folder} 中未找到图像文件")
        return
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 存储结果
    results = []
    
    # 处理每个图像
    for image_file in image_files:
        logger.info(f"\n处理图像: {os.path.basename(image_file)}")
        
        # 预处理图像
        image_tensor = preprocess_image(image_file, model_loader)
        if image_tensor is None:
            continue
        
        # 进行预测
        output = predict(model, image_tensor, device)
        if output is None:
            continue
        
        # 保存结果
        output_np = output.cpu().numpy().flatten()
        result = {
            'image': os.path.basename(image_file)
        }
        
        # 添加每个成分的预测值
        for i, val in enumerate(output_np):
            result[f'成分{i+1}'] = val
        
        results.append(result)
        logger.info(f"预测结果: {output_np}")
    
    # 如果没有结果，直接返回
    if not results:
        logger.info("没有成功预测的图像")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 保存到CSV
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"\n结果已保存到: {output_csv}")
    
    # 显示统计信息
    logger.info("\n统计信息:")
    logger.info(str(df.describe()))
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 为每个成分创建一个箱线图
    component_cols = [col for col in df.columns if col.startswith('成分')]
    df[component_cols].boxplot()
    
    plt.title('油菜籽成分含量预测统计')
    plt.ylabel('预测值')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/batch_prediction_stats.png')
    plt.show()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量测试油菜籽成分预测模型')
    parser.add_argument('--model', type=str, default='res/vanillanet.pt', help='模型路径')
    parser.add_argument('--folder', type=str, default='tests/test_images', help='测试图像文件夹')
    parser.add_argument('--output', type=str, default='results/prediction_results.csv', help='结果保存路径')
    args = parser.parse_args()
    
    # 设置模型路径和测试图像文件夹
    model_path = args.model
    image_folder = args.folder
    output_csv = args.output
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"错误: 模型文件不存在: {model_path}")
        exit()
    
    if not os.path.exists(image_folder):
        logger.error(f"错误: 图像文件夹不存在: {image_folder}")
        exit()
    
    # 运行批量测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_test(model_path, image_folder, output_csv, device)