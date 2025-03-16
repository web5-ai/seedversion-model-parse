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
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ModuleDict({
            'conv': nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, bias=True),
            'bn': nn.BatchNorm2d(out_channels)
        })
        self.attention = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        identity = x
        # 并行处理两个分支
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        # 合并分支
        x = x1 + x2
        
        # 注意力机制
        x = self.act['conv'](x)
        x = self.act['bn'](x)
        attention = self.attention(x)
        x = x * attention
        
        # 残差连接
        if identity.shape[1] == x.shape[1]:  # 检查通道数是否匹配
            x = x + identity
            
        return x

class FasterNet(nn.Module):
    def __init__(self, num_classes=2, channels=None):
        super().__init__()
        
        # 根据预训练权重的实际形状更新通道配置
        if channels is None:
            channels = [
                (512, 512),      # stage 0: 输入512，输出512
                (512, 1024),     # stage 1: 输入512，输出1024
                (1024, 4096),    # stage 2: 输入1024，输出4096 (关键修改)
                (4096, 1024),    # stage 3: 输入4096，输出1024
                (1024, 2048),    # stage 4: 输入1024，输出2048
                (2048, 2048),    # stage 5: 保持2048
                (2048, 2048),    # stage 6: 保持2048
                (2048, 2048),    # stage 7: 保持2048
                (2048, 2048),    # stage 8: 保持2048
                (2048, 4096),    # stage 9: 输出4096
                (4096, 4096)     # stage 10: 保持4096
            ]
        
        # 保存 channels 配置
        self.channels = channels
        
        # stem 层
        self.stem1 = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )
        
        # 修改 stem2 层结构
        self.stem2 = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1),
                nn.BatchNorm2d(512)
            ),
            'act': nn.ModuleDict({
                'conv': nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=3, bias=True),
                'bn': nn.BatchNorm2d(512)
            }),
            'attention': nn.Conv2d(512, 1, kernel_size=1)
        })
        self.stem2_bn = nn.BatchNorm2d(512)
        
        # stages
        self.stages = nn.ModuleList([
            StageBlock(in_ch, out_ch) for in_ch, out_ch in channels
        ])

        # 分类器
        final_channels = channels[-1][1]
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
        
        # 修改 stem2 的前向传播
        stem2_feat = self.stem2['conv'](x)
        act_out = self.stem2['act']['conv'](stem2_feat)
        act_out = self.stem2['act']['bn'](act_out)
        attention = self.stem2['attention'](act_out)
        x = stem2_feat * attention
        x = self.stem2_bn(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.avgpool(x)
        x = self.cls1(x)
        x = self.cls2(x)
        return x.flatten(1)

class ModelLoader:
    def __init__(self, model_path):
        # 配置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        try:
            self.logger.info(f"开始加载模型: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 记录预训练权重的键
            self.logger.info("预训练权重包含以下键:")
            for k in state_dict.keys():
                self.logger.info(f"- {k} (shape: {state_dict[k].shape})")

            if not self._validate_state_dict(state_dict):
                raise ValueError("无效的权重文件格式")

            if isinstance(state_dict, OrderedDict):
                self.logger.info("创建模型实例")
                # 在创建模型之前检测通道配置
                detected_channels = self._detect_channels_from_weights(state_dict)
                if detected_channels:
                    self.logger.info("使用从预训练权重检测到的通道配置")
                    base_model = FasterNet(num_classes=2, channels=detected_channels)
                else:
                    self.logger.info("使用默认通道配置")
                    base_model = FasterNet(num_classes=2)
                
                # 记录模型结构
                self.logger.info("模型结构:")
                for name, module in base_model.named_modules():
                    self.logger.info(f"- {name}: {module.__class__.__name__}")
                
                self.model = nn.Sequential(OrderedDict([
                    ('model', base_model)
                ])).to(self.device).eval()
                
                # 记录模型状态字典的键
                self.logger.info("模型状态字典包含以下键:")
                for k in self.model.state_dict().keys():
                    self.logger.info(f"- {k} (shape: {self.model.state_dict()[k].shape})")
                
                # 使用渐进式加载策略
                self._progressive_load(state_dict)
                
                if self._verify_model():
                    self.logger.info("模型加载完成并验证通过")
                else:
                    raise RuntimeError("模型验证失败")
                
            else:
                raise ValueError("加载的不是有效的权重字典")
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}", exc_info=True)
            self._handle_initialization_error(e)
            raise e

    def _progressive_load(self, state_dict):
        """渐进式加载权重"""
        try:
            self.logger.info("\n=== 开始渐进式加载 ===")
            
            # 预处理权重字典，移除前缀
            processed_dict = {}
            for k, v in state_dict.items():
                # 移除可能的前缀
                new_key = k.replace('module.', '').replace('model.', '')
                # 如果是 model 开头的键，添加回去
                if not new_key.startswith('model.'):
                    new_key = f'model.{new_key}'
                processed_dict[new_key] = v
            
            # 记录模型需要的键
            model_keys = set(self.model.state_dict().keys())
            
            # 尝试直接加载匹配的权重
            self.logger.info("尝试直接加载匹配的权重...")
            matched_dict = {k: v for k, v in processed_dict.items() if k in model_keys and v.shape == self.model.state_dict()[k].shape}
            self.model.load_state_dict(matched_dict, strict=False)
            
            # 处理形状不匹配的权重
            self.logger.info("处理形状不匹配的权重...")
            mismatched_keys = [k for k in processed_dict.keys() if k in model_keys and processed_dict[k].shape != self.model.state_dict()[k].shape]
            
            for k in mismatched_keys:
                try:
                    v = processed_dict[k]
                    target_shape = self.model.state_dict()[k].shape
                    
                    # 只处理维度相同的情况
                    if len(v.shape) == len(target_shape):
                        # 特殊处理卷积层权重
                        if len(v.shape) == 4:
                            # 创建新的权重张量
                            new_weight = torch.zeros(target_shape, device=v.device, dtype=v.dtype)
                            
                            # 复制可以复制的部分
                            min_out = min(v.shape[0], target_shape[0])
                            min_in = min(v.shape[1], target_shape[1])
                            min_h = min(v.shape[2], target_shape[2])
                            min_w = min(v.shape[3], target_shape[3])
                            
                            new_weight[:min_out, :min_in, :min_h, :min_w] = v[:min_out, :min_in, :min_h, :min_w]
                        else:
                            # 处理其他类型的权重
                            new_weight = torch.zeros(target_shape, device=v.device, dtype=v.dtype)
                            slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(v.shape, target_shape))
                            new_weight[slices] = v[slices]
                        
                        # 加载调整后的权重
                        temp_dict = {k: new_weight}
                        self.model.load_state_dict(temp_dict, strict=False)
                        self.logger.info(f"已调整并加载权重: {k}, 从 {v.shape} 到 {target_shape}")
                except Exception as e:
                    self.logger.warning(f"处理参数 {k} 时出错: {str(e)}")
            
            # 验证加载结果
            loaded_params = sum(1 for name, param in self.model.named_parameters() 
                          if not torch.any(torch.isnan(param)))
            total_params = len(list(self.model.parameters()))
            
            self.logger.info(f"\n成功加载 {loaded_params}/{total_params} 个参数")
            
            return loaded_params > 0
            
        except Exception as e:
            self.logger.error(f"加载权重时出错: {str(e)}", exc_info=True)
            return False

    def _validate_state_dict(self, state_dict):
        """验证权重字典的有效性"""
        try:
            # 检查基本结构
            if not isinstance(state_dict, (OrderedDict, dict)):
                return False
                
            # 检查关键层是否存在
            essential_layers = ['stem1', 'stem2', 'stages', 'cls1']
            for layer in essential_layers:
                if not any(layer in k for k in state_dict.keys()):
                    print(f"警告: 未找到 {layer} 层的权重")
                    return False
                    
            # 检查权重值的有效性
            for k, v in state_dict.items():
                if not isinstance(v, torch.Tensor):
                    print(f"警告: {k} 的值不是张量")
                    return False
                if v.isnan().any():
                    print(f"警告: {k} 包含 NaN 值")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"验证权重字典时出错: {str(e)}")
            return False

    def _verify_model(self):
        """验证加载后的模型"""
        try:
            # 检查模型是否在正确的设备上
            if next(self.model.parameters()).device != self.device:
                print("警告: 模型不在指定设备上")
                return False
                
            # 检查模型是否处于评估模式
            if self.model.training:
                print("警告: 模型不在评估模式")
                return False
                
            # 进行简单的前向传播测试
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                try:
                    # 获取模型的第一层
                    first_layer = self.model.model.stem1[0]
                    # 获取第一层的输出通道数
                    out_channels = first_layer.out_channels
                    
                    # 确保输入图像的尺寸正确
                    if dummy_input.shape[2] != 224 or dummy_input.shape[3] != 224:
                        dummy_input = torch.nn.functional.interpolate(
                            dummy_input, size=(224, 224), mode='bilinear', align_corners=False
                        )
                    
                    # 执行前向传播
                    output = self.model(dummy_input)
                    
                    # 检查输出形状
                    if output.shape[0] != 1 or output.shape[1] != 2:
                        print(f"警告: 输出形状不正确: {output.shape}")
                        return False
                        
                except Exception as e:
                    print(f"前向传播测试失败: {str(e)}")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"验证模型时出错: {str(e)}")
            return False

    def _adjust_attention_weights(self, key, weight):
        """调整注意力层权重"""
        try:
            if 'act.weight' in key:
                return weight.mean(dim=1, keepdim=True)
            elif 'attention' in key:
                return weight.transpose(0, 1)
            return weight
        except:
            print(f"警告: 无法调整权重 {key}")
            return None

    def _adjust_layer_weights(self, key, weight):
        """调整其他层权重"""
        try:
            target_shape = self.model.state_dict()[key].shape
            if weight.shape != target_shape:
                if len(weight.shape) == len(target_shape):
                    # 处理通道数不匹配
                    if weight.shape[0] < target_shape[0]:
                        weight = torch.cat([weight] * (target_shape[0] // weight.shape[0]), dim=0)
                    else:
                        weight = weight[:target_shape[0]]
            return weight
        except:
            print(f"警告: 无法调整权重 {key}")
            return None

    def _load_state_dict(self, state_dict):
        """加载权重并返回缺失和多余的键"""
        """加载权重并返回缺失和多余的键"""
        try:
            # 预处理权重字典
            processed_dict = {}
            for k, v in state_dict.items():
                # 移除可能的前缀
                new_key = k.replace('module.', '').replace('model.', '')
                # 如果是 model 开头的键，添加回去
                if not new_key.startswith('model.'):
                    new_key = f'model.{new_key}'
                processed_dict[new_key] = v
            
            # 尝试加载权重
            incompatible_keys = self.model.load_state_dict(processed_dict, strict=False)
            
            # 记录加载结果
            if incompatible_keys.missing_keys:
                self.logger.info("\n=== 未加载的参数 ===")
                for k in incompatible_keys.missing_keys:
                    self.logger.info(f"缺失: {k}")
            
            if incompatible_keys.unexpected_keys:
                self.logger.info("\n=== 多余的参数 ===")
                for k in incompatible_keys.unexpected_keys:
                    self.logger.info(f"多余: {k}")
            
            return incompatible_keys.missing_keys, incompatible_keys.unexpected_keys
            
        except Exception as e:
            self.logger.error(f"加载权重时出错: {str(e)}")
            raise e

    def _handle_loading_results(self, missing_keys, unexpected_keys):
        """处理权重加载结果"""
        if missing_keys:
            print("\n=== 未加载的模型参数 ===")
            for k in missing_keys:
                print(f"- {k}")
                if k in self.model.state_dict():
                    print(f"  当前形状: {self.model.state_dict()[k].shape}")
        
        if unexpected_keys:
            print("\n=== 预训练模型中多余的参数 ===")
            for k in unexpected_keys:
                print(f"- {k}")

    def _handle_loading_error(self, error, state_dict):
        """处理权重加载错误"""
        print("\n=== 加载失败详细信息 ===")
        print(str(error))
        
        print("\n=== 形状不匹配的参数 ===")
        for k in state_dict.keys():
            if k in self.model.state_dict():
                pretrained_shape = state_dict[k].shape
                model_shape = self.model.state_dict()[k].shape
                if pretrained_shape != model_shape:
                    print(f"\n参数: {k}")
                    print(f"预训练形状: {pretrained_shape}")
                    print(f"模型形状: {model_shape}")

    def _handle_initialization_error(self, error):
        """处理初始化错误"""
        logging.error(f"模型初始化失败: {str(error)}")
        if self.model is not None:
            try:
                self.model = None
            except:
                pass

    # 在 ModelLoader 类的末尾添加
    def get_transform(self):
        """获取图像预处理转换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),          # 转换为张量
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 标准化参数
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image):
        """预处理单张图像"""
        transform = self.get_transform()
        return transform(image).unsqueeze(0).to(self.device)  # 添加 batch 维度并移至正确设备

    def _detect_channels_from_weights(self, state_dict):
        """从预训练权重中检测通道配置"""
        try:
            channels = []
            stage_keys = [k for k in state_dict.keys() if 'stages' in k and 'conv1.0.weight' in k]
            stage_keys.sort()  # 确保按顺序处理
            
            for key in stage_keys:
                weight = state_dict[key]
                in_channels = weight.shape[1]
                out_channels = weight.shape[0]
                channels.append((in_channels, out_channels))
                self.logger.info(f"检测到通道配置: {key} -> ({in_channels}, {out_channels})")
            
            return channels if channels else None
        except Exception as e:
            self.logger.warning(f"从权重检测通道配置失败: {str(e)}")
            return None