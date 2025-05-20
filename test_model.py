"""油菜籽成分预测模型测试工具"""

import os
import sys
import argparse
import logging
import torch
import pandas as pd
from PIL import Image
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils.model_loader import ModelLoader
from config import MODEL_CONFIG, IMAGE_CONFIG, SYSTEM_CONFIG

def setup_logger(name="ModelTest", level=None):
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

def get_available_models(models_dir="weights") -> List[Dict[str, str]]:
    if not os.path.exists(models_dir):
        logger.error(f"模型目录不存在: {models_dir}")
        return []

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]

    if not model_files:
        logger.warning(f"在 {models_dir} 目录中未找到模型文件")
        return []

    models = []
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        models.append({
            "path": model_path,
            "name": model_name
        })

    return models

def test_single_image(model_loader: ModelLoader, image_path: str) -> Dict[str, Union[str, float]]:
    try:
        if not os.path.exists(image_path):
            logger.error(f"图像不存在: {image_path}")
            return {}

        image = Image.open(image_path).convert('RGB')
        size = 256 if model_loader.model_name == 'Swin' else 224
        image_tensor = model_loader.preprocess_image(image, size)

        output = model_loader.predict(image_tensor)
        output_np = output.cpu().numpy().flatten()

        if len(output_np) < 2:
            logger.error(f"输出维度错误，应至少为2，实际为{len(output_np)}")
            return {}

        result = {
            'image': os.path.basename(image_path),
            'model': model_loader.model_name,
            'oil': float(output_np[0]),
            'protein': float(output_np[1]),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info(f"预测结果 - 图像: {os.path.basename(image_path)}")
        logger.info(f"  模型: {model_loader.model_name}")
        logger.info(f"  油脂: {result['oil']:.4f}")
        logger.info(f"  蛋白质: {result['protein']:.4f}")

        return result

    except Exception as e:
        logger.error(f"测试图像时出错: {str(e)}")
        return {}

def test_batch_images(model_loader: ModelLoader, images_dir: str, output_csv: str = None) -> List[Dict]:
    if not os.path.exists(images_dir):
        logger.error(f"图像目录不存在: {images_dir}")
        return []

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(Path(images_dir).glob(f"*{ext}")))
        image_files.extend(list(Path(images_dir).glob(f"*{ext.upper()}")))

    if not image_files:
        logger.warning(f"在 {images_dir} 目录中未找到图像文件")
        return []

    logger.info(f"找到 {len(image_files)} 个图像文件")

    os.makedirs("results", exist_ok=True)

    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_csv = f"results/batch_test_{model_loader.model_name}_{timestamp}.csv"

    results = []

    def process_image(image_path):
        result = test_single_image(model_loader, str(image_path))
        if result:
            results.append(result)

    with ThreadPoolExecutor() as executor:
        list(executor.map(process_image, image_files))

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"批量测试结果已保存至: {output_csv}")

    return results

def test_multiple_models(image_path: str, models_dir: str = "weights", except_models: List[str] = None) -> List[Dict]:
    if not os.path.exists(image_path):
        logger.error(f"图像不存在: {image_path}")
        return []

    models = get_available_models(models_dir)

    if not models:
        return []

    if except_models:
        models = [m for m in models if m["name"] not in except_models]

    logger.info(f"将使用 {len(models)} 个模型进行测试")

    results = []

    for model_info in models:
        logger.info(f"正在测试模型: {model_info['name']}")

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_loader = ModelLoader(model_info["path"], debug=False, device=device)

            model_loader.load_model(model_info["name"])
            result = test_single_image(model_loader, image_path)

            if result:
                results.append(result)

            model_loader.unload_model()

        except Exception as e:
            logger.error(f"测试模型 {model_info['name']} 时出错: {str(e)}")

    if results:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_csv = f"results/multi_model_test_{timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"多模型测试结果已保存至: {output_csv}")

    return results

def main():
    parser = argparse.ArgumentParser(description='油菜籽成分预测模型测试工具')

    subparsers = parser.add_subparsers(dest='command', help='命令')

    single_parser = subparsers.add_parser('single', help='单模型单图像测试')
    single_parser.add_argument('--model', type=str, default=MODEL_CONFIG["default_model"], help='模型路径或名称')
    single_parser.add_argument('--image', type=str, default=IMAGE_CONFIG["default_image_path"], help='图像路径')

    batch_parser = subparsers.add_parser('batch', help='单模型批量图像测试')
    batch_parser.add_argument('--model', type=str, default=MODEL_CONFIG["default_model"], help='模型路径或名称')
    batch_parser.add_argument('--dir', type=str, default=IMAGE_CONFIG["default_images_dir"], help='图像目录')
    batch_parser.add_argument('--output', type=str, help='输出CSV文件路径')

    multi_parser = subparsers.add_parser('multi', help='多模型测试')
    multi_parser.add_argument('--image', type=str, default=IMAGE_CONFIG["default_image_path"], help='图像路径')
    multi_parser.add_argument('--except', type=str, nargs='*', help='排除的模型名称')

    subparsers.add_parser('list', help='检查可用模型')

    parser.add_argument('--seed', type=int, default=SYSTEM_CONFIG["default_seed"], help='随机种子')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='运行设备')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"使用设备: {device}")

    if args.command == 'single':
        model_path = args.model
        if not os.path.exists(model_path) and not model_path.endswith('.pt'):
            model_path = os.path.join('weights', f"{model_path}.pt")

        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_loader = ModelLoader(model_path, debug=False, device=device)
        model_loader.load_model(model_name)

        test_single_image(model_loader, args.image)

    elif args.command == 'batch':
        model_path = args.model
        if not os.path.exists(model_path) and not model_path.endswith('.pt'):
            model_path = os.path.join('weights', f"{model_path}.pt")

        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_loader = ModelLoader(model_path, debug=False, device=device)
        model_loader.load_model(model_name)

        test_batch_images(model_loader, args.dir, args.output)

    elif args.command == 'multi':
        test_multiple_models(args.image, except_models=args)

    elif args.command == 'list':
        models = get_available_models()

        if models:
            logger.info(f"找到 {len(models)} 个可用模型:")
            for i, model in enumerate(models, 1):
                logger.info(f"{i}. {model['name']} - {model['path']}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
