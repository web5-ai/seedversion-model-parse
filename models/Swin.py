from .model_zoo import Efficient_net,FasterNet,init_weights,SwinTransformerV2,mpvit_base,vanillanet_13_x1_5,vanillanet_5,vanillanet_13
import torch
from torch import nn
from .swinv2_config import get_config
from .swinv2_config_large import get_config
import pdb

def load_pretrained(model, weight):
    checkpoint = torch.load(weight, map_location='cpu')
    # state_dict = checkpoint['model']
    # 把model.去掉，然后对模型结构按照字典进行更改
    # 主要修改图像尺寸，窗口大小
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=True)
    return state_dict
    # # delete relative_position_index since we always re-init it
    # relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    # for k in relative_position_index_keys:
    #     del state_dict[k]

    # # delete relative_coords_table since we always re-init it
    # relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    # for k in relative_position_index_keys:
    #     del state_dict[k]

    # # delete attn_mask since we always re-init it
    # attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    # for k in attn_mask_keys:
    #     del state_dict[k]

    # # bicubic interpolate relative_position_bias_table if not match
    # relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    # for k in relative_position_bias_table_keys:
    #     relative_position_bias_table_pretrained = state_dict[k]
    #     relative_position_bias_table_current = model.state_dict()[k]
    #     L1, nH1 = relative_position_bias_table_pretrained.size()
    #     L2, nH2 = relative_position_bias_table_current.size()
    #     if nH1 != nH2:
    #         print(f"Error in loading {k}, passing......")
    #     else:
    #         if L1 != L2:
    #             # bicubic interpolate relative_position_bias_table if not match
    #             S1 = int(L1 ** 0.5)
    #             S2 = int(L2 ** 0.5)
    #             relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
    #                 relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
    #                 mode='bicubic')
    #             state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # # bicubic interpolate absolute_pos_embed if not match
    # absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    # for k in absolute_pos_embed_keys:
    #     # dpe
    #     absolute_pos_embed_pretrained = state_dict[k]
    #     absolute_pos_embed_current = model.state_dict()[k]
    #     _, L1, C1 = absolute_pos_embed_pretrained.size()
    #     _, L2, C2 = absolute_pos_embed_current.size()
    #     if C1 != C1:
    #         print(f"Error in loading {k}, passing......")
    #     else:
    #         if L1 != L2:
    #             S1 = int(L1 ** 0.5)
    #             S2 = int(L2 ** 0.5)
    #             absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
    #             absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
    #             absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
    #                 absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
    #             absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
    #             absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
    #             state_dict[k] = absolute_pos_embed_pretrained_resized

    # # check classifier, if not match, then re-init classifier to zero
    # head_bias_pretrained = state_dict['head.bias']
    # Nc1 = head_bias_pretrained.shape[0]
    # Nc2 = model.head.bias.shape[0]
    # if (Nc1 != Nc2):
    #     if Nc1 == 21841 and Nc2 == 1000:
    #         print("loading ImageNet-22K weight to ImageNet-1K ......")
    #         map22kto1k_path = f'map22kto1k.txt'
    #         with open(map22kto1k_path) as f:
    #             map22kto1k = f.readlines()
    #         map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
    #         state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
    #         state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
    #     else:
    #         torch.nn.init.constant_(model.head.bias, 0.)
    #         torch.nn.init.constant_(model.head.weight, 0.)
    #         del state_dict['head.weight']
    #         del state_dict['head.bias']
    #         print(f"Error in loading classifier head, re-init classifier head to 0")

    # msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # print("=> loaded successfully")

    # del checkpoint
    # torch.cuda.empty_cache()

class SwinV2(nn.Module):
    def __init__(self, number_of_classes = 2, device = 'cpu'):
        self.device = device
        super(SwinV2, self).__init__()
        config = get_config(args={})
        self.model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
        
        # self.model.load_state_dict(torch.load(weight)['model'], strict=False)

        self.model.head = nn.Linear(self.model.num_features, number_of_classes) if number_of_classes > 0 else nn.Identity()
    
    def func_transition(self,outputs):
        # outputs = outputs.view(-1)

        outputs[:,0] = (torch.sigmoid(outputs[:,0]) * 100)
        outputs[:,1] = (torch.sigmoid(outputs[:,1]) * 100)
        return outputs[:, :2]
        
    def forward(self, x):
        head_out = self.model(x)
        out = self.func_transition(head_out)
        out = out[0]
        out[1] = out[1] * (29.1-17.4) / 100 + 17.4
        out[0] = out[0] * (50.5 - 32.5) / 100 + 32.5
        return out
    
    def load_model_weight(self, weight_path):
        checkpoint = torch.load(weight_path, map_location=self.device)

        state_dict = {k.replace("model.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(state_dict, strict=True)
        return self.model.state_dict()