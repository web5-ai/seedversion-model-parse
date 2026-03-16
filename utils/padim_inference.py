# AI-GENERATED-START
import os
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_padim_image(image_input: Image.Image | str, image_size: int = 256) -> torch.Tensor:
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, str) and image_input.startswith(("http://", "https://")):
        response = requests.get(image_input, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"图像文件不存在: {image_input}")
        image = Image.open(image_input).convert("RGB")
    else:
        raise TypeError(f"不支持的图像输入类型: {type(image_input)}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


class PaDiMInference:
    def __init__(self, model_path: str, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PaDiM模型文件不存在: {model_path}")

        self.device = device
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

        self.mean = {key: _to_numpy(value) for key, value in checkpoint["mean"].items()}
        self.inv_covariance = {
            key: _to_numpy(value) for key, value in checkpoint["inv_covariance"].items()
        }
        self.random_projectors = {
            key: _to_numpy(value) for key, value in checkpoint["random_projectors"].items()
        }
        self.threshold = float(checkpoint["threshold"])
        self.layers = checkpoint.get("layers", ["layer2", "layer3"])
        self.embedding_dim = checkpoint["embedding_dim"]
        self.layer_to_idx = {"layer2": 5, "layer3": 6}

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2]).to(device)
        self.feature_extractor.eval()

        self.features: dict[str, torch.Tensor] = {}
        for name in self.layers:
            idx = self.layer_to_idx[name]
            self.feature_extractor[idx].register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            self.features[name] = output

        return hook

    def predict(self, image_tensor: torch.Tensor) -> float:
        image_tensor = image_tensor.to(self.device)
        self.features.clear()

        with torch.no_grad():
            _ = self.feature_extractor(image_tensor)

        scores_per_layer = []
        for layer in self.layers:
            feat = self.features[layer]
            _, channels, height, width = feat.shape
            feat_flat = feat.squeeze(0).permute(1, 2, 0).reshape(-1, channels).cpu().numpy()
            feat_flat = np.ascontiguousarray(np.nan_to_num(feat_flat, nan=0.0, posinf=1e6, neginf=-1e6), dtype=np.float64)

            projector = np.ascontiguousarray(self.random_projectors[layer], dtype=np.float64)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                projected_feature = np.nan_to_num(
                    feat_flat @ projector, nan=0.0, posinf=1e6, neginf=-1e6
                )
            mean = np.ascontiguousarray(self.mean[layer], dtype=np.float64)
            inv_cov = np.ascontiguousarray(self.inv_covariance[layer], dtype=np.float64)

            mahal_scores = []
            for index in range(height * width):
                diff = projected_feature[index] - mean[index]
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    score = diff @ inv_cov[index] @ diff.T
                mahal_scores.append(float(np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)))
            scores_per_layer.append(float(np.max(mahal_scores)))

        return float(np.max(scores_per_layer))

    def infer(
        self,
        image_input: Image.Image | str,
        threshold_buffer_a: int,
        threshold_buffer_b: int,
    ) -> dict:
        image_tensor = load_padim_image(image_input)
        score = self.predict(image_tensor)

        if score < self.threshold - threshold_buffer_a:
            is_normal = 1
            label = "rapeseed"
            is_rapeseed = True
        elif score > self.threshold + threshold_buffer_b:
            is_normal = -1
            label = "non_rapeseed"
            is_rapeseed = False
        else:
            is_normal = 0
            label = "uncertain"
            is_rapeseed = None

        return {
            "score": score,
            "is_normal": is_normal,
            "is_rapeseed": is_rapeseed,
            "label": label,
            "threshold": self.threshold,
            "threshold_buffer_a": threshold_buffer_a,
            "threshold_buffer_b": threshold_buffer_b,
        }


def run_padim_inference(
    image_input: Image.Image | str,
    model_path: str,
    threshold_buffer_a: int,
    threshold_buffer_b: int,
    device: str | None = None,
) -> dict:
    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    padim = PaDiMInference(model_path=model_path, device=runtime_device)
    result = padim.infer(
        image_input=image_input,
        threshold_buffer_a=threshold_buffer_a,
        threshold_buffer_b=threshold_buffer_b,
    )
    result["device"] = runtime_device
    return result
# AI-GENERATED-END
