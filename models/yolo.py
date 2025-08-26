from ultralytics import YOLO
import torch
import os

class yolo_model:
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化YOLO模型，支持.pt和.onnx格式

        Args:
            model_path: 模型文件路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.model_path = model_path

        # 检查模型文件格式
        if model_path.endswith('.onnx'):
            # ONNX模型加载
            print(f"加载ONNX模型: {model_path}")
            self.model = YOLO(model_path, task='detect')
            # ONNX模型通常在推理时指定设备
            self.is_onnx = True
        else:
            # PyTorch模型加载
            print(f"加载PyTorch模型: {model_path}")
            self.model = YOLO(model_path)
            self.is_onnx = False

            # 将PyTorch模型移动到指定设备
            if hasattr(self.model, 'model') and self.model.model is not None:
                self.model.model.to(device)
    def detect(self, image, conf_threshold=0.9, iou_threshold=0.5):
        """
        使用YOLO模型进行目标检测

        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        if self.is_onnx:
            # ONNX模型检测
            # ONNX模型在CPU上运行更稳定
            results = self.model(image, device='cpu', conf=conf_threshold, iou=iou_threshold)
        else:
            # PyTorch模型检测
            # 确保模型在正确的设备上
            if hasattr(self.model, 'model') and self.model.model is not None:
                self.model.model.to(self.device)

            # 进行检测，使用指定的设备和阈值参数
            results = self.model(image, device=self.device, conf=conf_threshold, iou=iou_threshold)

        return results
