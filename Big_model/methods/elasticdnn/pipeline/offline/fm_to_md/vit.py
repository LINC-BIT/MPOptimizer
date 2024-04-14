
import torch
from torch import nn 
from copy import deepcopy

from .base import FM_to_MD_Util
from utils.common.log import logger
from utils.dl.common.model import set_module, get_module, get_super_module


class FM_to_MD_ViT_Util(FM_to_MD_Util):
    def init_md_from_fm_by_reducing_width(self, fm: nn.Module, reducing_width_ratio: int) -> nn.Module:
        fm_vit = deepcopy(fm)
        
        def _f(n):
            return int(n // reducing_width_ratio)
        
        # def _rand_indexes(n):
            # return torch.randperm(n)[0: int(n // reducing_width_ratio)]
            
        def l1_max_indexes(p: torch.Tensor, dim=0):
            assert dim in [0, 1]
            assert p.dim() in [1, 2, 4]
            
            if dim == 1:
                p = p.T
            
            p_norm = p.abs().contiguous().view(p.size(0), -1).sum(dim=1)
            n = p.size(0)
            return p_norm.argsort(descending=True)[0: int(n // reducing_width_ratio)].sort()[0]
        
        # first_attn = True
        
        for block_i, block in enumerate(fm_vit.blocks):
            qkv = block.attn.qkv
            
            new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                qkv.bias is not None, qkv.weight.device)
            indexes = l1_max_indexes(qkv.weight.data, 0)
            
            new_qkv.weight.data.copy_(qkv.weight.data[indexes])
            if qkv.bias is not None:
                new_qkv.bias.data.copy_(qkv.bias.data[indexes])
            set_module(fm_vit, f'blocks.{block_i}.attn.qkv', new_qkv)

            proj = block.attn.proj
            new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
                                proj.bias is not None, proj.weight.device)
            new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
            if proj.bias is not None:
                new_proj.bias.data.copy_(proj.bias.data)
            set_module(fm_vit, f'blocks.{block_i}.attn.proj', new_proj)
            
            fc1 = block.mlp.fc1
            new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
                                fc1.bias is not None, fc1.weight.device)
            indexes = l1_max_indexes(fc1.weight.data, 0)
            new_fc1.weight.data.copy_(fc1.weight.data[indexes])
            if fc1.bias is not None:
                new_fc1.bias.data.copy_(fc1.bias.data[indexes])
            set_module(fm_vit, f'blocks.{block_i}.mlp.fc1', new_fc1)

            fc2 = block.mlp.fc2
            new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
                                fc2.bias is not None, fc2.weight.device)
            new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
            if fc2.bias is not None:
                new_fc2.bias.data.copy_(fc2.bias.data)
            set_module(fm_vit, f'blocks.{block_i}.mlp.fc2', new_fc2)
            # reduce dim_embedding
            # if name.endswith('patch_embed.proj'):
            #     continue
                
            #     new_layer = nn.Conv2d(module.in_channels, _f(module.out_channels), module.kernel_size, module.stride,
            #                          module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode,
            #                          module.weight.device)
                
            #     rand_indexes = l1_max_indexes(module.weight.data)
            #     new_layer.weight.data.copy_(module.weight.data[rand_indexes])
            #     if new_layer.bias is not None:
            #         new_layer.bias.data.copy_(module.bias.data[rand_indexes])
                
            #     fm_vit.cls_token.data = fm_vit.cls_token.data[:, :, rand_indexes]
            #     fm_vit.pos_embed.data = fm_vit.pos_embed.data[:, :, rand_indexes]
            
            # elif isinstance(module, nn.Linear):
                
            #     if 'head' in name:
            #         continue
                
            #         new_layer = nn.Linear(_f(module.in_features), module.out_features, 
            #                             module.bias is not None, module.weight.device)
            #         new_layer.weight.data.copy_(module.weight.data[:, l1_max_indexes(module.weight.data, 1)])
            #         if new_layer.bias is not None:
            #             new_layer.bias.data.copy_(module.bias.data)
            #     else:
            #         first_attn = False
            #         if first_attn:
            #             first_attn = False
            #             new_layer = nn.Linear(module.in_features, _f(module.out_features), 
            #                                 module.bias is not None, module.weight.device)
                        
            #             rand_indexes = l1_max_indexes(module.weight.data)
            #             new_layer.weight.data.copy_(module.weight.data[rand_indexes])
            #             if new_layer.bias is not None:
            #                 new_layer.bias.data.copy_(module.bias.data[rand_indexes])
            #         else:
            #             new_layer = nn.Linear(_f(module.in_features), _f(module.out_features), 
            #                                 module.bias is not None, module.weight.device)
                        
            #             rand_indexes = l1_max_indexes(module.weight.data)
            #             new_layer.weight.data.copy_(module.weight.data[rand_indexes][:, l1_max_indexes(module.weight.data, 1)])
            #             if new_layer.bias is not None:
            #                 new_layer.bias.data.copy_(module.bias.data[rand_indexes])
                        
            # elif isinstance(module, nn.LayerNorm) and ('block' in name or name == 'norm' or name == 'norm.0'):
            #     new_layer = nn.LayerNorm(_f(module.normalized_shape[0]), eps=module.eps, device=module.weight.device)
            #     rand_indexes = l1_max_indexes(module.weight.data)
            #     new_layer.weight.data.copy_(module.weight.data[rand_indexes])
            #     new_layer.bias.data.copy_(module.bias.data[rand_indexes])
                
            # else:
            #     continue
            
            # original_layer_str = str(module)
            # set_module(fm_vit, name, new_layer)
            # logger.debug(f'set_module, {name}, {new_layer}')
            # logger.debug(f'slim {name} from {original_layer_str} to {new_layer}')
        
        return fm_vit