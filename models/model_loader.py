import torch
import torch.nn as nn
import logging
from collections import OrderedDict
from torchvision import transforms  # 添加缺失的导入

class StageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        # 修改 act 层结构以匹配预训练模型
        self.act = nn.Conv2d(1, out_channels, kernel_size=7, padding=3, bias=True)
        self.act_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        # 先将特征压缩到单通道
        b, c, h, w = x.shape
        x_pooled = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        x = self.act(x_pooled)  # [B, C, H, W]
        x = self.act_bn(x)
        return x + identity

class FasterNet(nn.Module):
    def __init__(self, num_classes=2, channels=None):
        super().__init__()
        
        self.stem1 = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )
        
        self.stem2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1, kernel_size=7, stride=1, padding=3, bias=True)
        )
        self.stem2_bn = nn.BatchNorm2d(512)

        # 修改通道配置以匹配预训练模型
        if channels is None:
            channels = [
                (512, 512),    # stage 0
                (512, 1024),   # stage 1
                (1024, 2048),  # stage 2
                (2048, 2048),  # stage 3
                (2048, 2048),  # stage 4
                (2048, 2048),  # stage 5
                (2048, 2048),  # stage 6
                (2048, 2048),  # stage 7
                (2048, 2048),  # stage 8
                (2048, 4096),  # stage 9
                (4096, 4096)   # stage 10
            ]
        
        self.stages = nn.ModuleList([
            StageBlock(in_ch, out_ch) for in_ch, out_ch in channels
        ])

        # 获取最后一层的输出通道数
        final_channels = channels[-1][1] if channels else 4096
        
        # 修改分类器结构以匹配预训练模型
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls1 = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=1),
            nn.BatchNorm2d(final_channels),
            nn.Conv2d(final_channels, num_classes, kernel_size=1)
        )
        self.cls2 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.stem1(x)
        stem2_out = self.stem2(x)  # [B, 1, H, W]
        x = stem2_out.expand(-1, x.size(1), -1, -1)  # 扩展到与输入相同的通道数
        x = self.stem2_bn(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.avgpool(x)
        x = self.cls1(x)
        x = self.cls2(x)
        return x.flatten(1)

class ModelLoader:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            if isinstance(state_dict, OrderedDict):
                # 创建模型
                self.model = FasterNet(num_classes=2).to(self.device)
                
                # 分析我们的模型结构
                print("\n=== 我们的模型结构 ===")
                for name, param in self.model.named_parameters():
                    print(f"参数名: {name}, 形状: {param.shape}")
                
                print("\n=== 预训练权重结构 ===")
                for k, v in state_dict.items():
                    print(f"权重名: {k}, 形状: {v.shape}")
                
                # 加载权重
                try:
                    self.model.load_state_dict(state_dict)
                except RuntimeError as e:
                    logging.warning(f"直接加载失败: {str(e)}")
                    print("\n=== 开始权重调整 ===")
                    adjusted_state_dict = self._adjust_state_dict(state_dict)
                    try:
                        self.model.load_state_dict(adjusted_state_dict, strict=False)
                        print("\n=== 成功加载的权重 ===")
                        loaded_keys = set(self.model.state_dict().keys())
                        for k in adjusted_state_dict.keys():
                            if k in loaded_keys:
                                print(f"成功: {k}")
                            else:
                                print(f"未加载: {k}")
                    except RuntimeError as e:
                        logging.error(f"调整后加载仍然失败: {str(e)}")
                        raise
                
                self.model.eval()
                logging.info("模型加载成功")
            else:
                self.model = state_dict
                self.model.eval()
                
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            raise e

    def _adjust_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        our_state_dict = self.model.state_dict()
        
        print("\n=== 权重映射分析 ===")
        for our_key, our_param in our_state_dict.items():
            print(f"\n处理参数: {our_key}, 目标形状: {our_param.shape}")
            
            # 尝试找到对应的预训练权重
            pretrained_key = our_key
            if 'model.' + our_key in state_dict:
                pretrained_key = 'model.' + our_key
            
            if pretrained_key in state_dict:
                v = state_dict[pretrained_key]
                print(f"找到预训练权重: {pretrained_key}, 形状: {v.shape}")
                
                # 检查形状是否匹配
                if v.shape != our_param.shape:
                    print(f"形状不匹配: 预期 {our_param.shape}, 实际 {v.shape}")
                    # 如果是 act 层，保持原样
                    if 'act' in our_key:
                        print("保持 act 层权重原样")
                    else:
                        print("跳过不匹配的权重")
                        continue
                new_state_dict[our_key] = v
            else:
                print(f"未找到对应的预训练权重")
        
        return new_state_dict