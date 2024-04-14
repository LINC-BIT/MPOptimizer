from copy import deepcopy
from typing import Optional, Union
import torch
from torch import nn 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import tqdm

from utils.dl.common.model import LayerActivation, get_model_device, get_model_size, set_module
from .base import Abs, KTakesAll, ElasticDNNUtil, Layer_WrappedWithFBS
from utils.common.log import logger


class SqueezeLast(nn.Module):
    def __init__(self):
        super(SqueezeLast, self).__init__()
    
    def forward(self, x):
        return x.squeeze(-1)
    
    
class ProjConv_WrappedWithFBS(Layer_WrappedWithFBS):
    def __init__(self, raw_conv2d: nn.Conv2d, r):
        super(ProjConv_WrappedWithFBS, self).__init__()
        
        self.fbs = nn.Sequential(
            Abs(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(raw_conv2d.in_channels, raw_conv2d.out_channels // r),
            nn.ReLU(),
            nn.Linear(raw_conv2d.out_channels // r, raw_conv2d.out_channels),
            nn.ReLU()
        )
        
        self.raw_conv2d = raw_conv2d
        # self.raw_bn = raw_bn # remember clear the original BNs in the network
        
        nn.init.constant_(self.fbs[5].bias, 1.)
        nn.init.kaiming_normal_(self.fbs[5].weight)

    def forward(self, x):
        raw_x = self.raw_conv2d(x)
        
        if self.use_cached_channel_attention and self.cached_channel_attention is not None:
            channel_attention = self.cached_channel_attention
        else:
            self.cached_raw_channel_attention = self.fbs(x)
            self.cached_channel_attention = self.k_takes_all(self.cached_raw_channel_attention)
            
            channel_attention = self.cached_channel_attention
        
        return raw_x * channel_attention.unsqueeze(2).unsqueeze(3)


class Linear_WrappedWithFBS(Layer_WrappedWithFBS):
    def __init__(self, linear: nn.Linear, r):
        super(Linear_WrappedWithFBS, self).__init__()
        
        self.linear = linear
        
        # for conv: (B, C_in, H, W) -> (B, C_in) -> (B, C_out)
        # for mlp in ViT: (B, #patches, D: dim of patches embedding) -> (B, D) -> (B, C_out)
        self.fbs = nn.Sequential(
            Rearrange('b n d -> b d n'),
            Abs(),
            nn.AdaptiveAvgPool1d(1),
            SqueezeLast(),
            nn.Linear(linear.in_features, linear.out_features // r),
            nn.ReLU(),
            nn.Linear(linear.out_features // r, linear.out_features),
            nn.ReLU()
        )
        
        nn.init.constant_(self.fbs[6].bias, 1.)
        nn.init.kaiming_normal_(self.fbs[6].weight)
        
    
    def forward(self, x):
        if self.use_cached_channel_attention and self.cached_channel_attention is not None:
            channel_attention = self.cached_channel_attention
        else:
            self.cached_raw_channel_attention = self.fbs(x)
            self.cached_channel_attention = self.k_takes_all(self.cached_raw_channel_attention)
            
            channel_attention = self.cached_channel_attention
        
        raw_res = self.linear(x)
        
        return channel_attention.unsqueeze(1) * raw_res
    
    
# class ToQKV_WrappedWithFBS(Layer_WrappedWithFBS):
#     """
#     This regards to_q/to_k/to_v as a whole (in fact it consists of multiple heads) and prunes it.
#     It seems different channels of different heads are pruned according to the input. 
#     This is different from "removing some head" or "removing the same channels in each head".
#     """
#     def __init__(self, to_qkv: nn.Linear, r):
#         super(ToQKV_WrappedWithFBS, self).__init__()
        
#         # self.to_qkv = to_qkv
        
#         self.to_qk = nn.Linear(to_qkv.in_features, to_qkv.out_features // 3 * 2, bias=to_qkv.bias is not None)
#         self.to_v = nn.Linear(to_qkv.in_features, to_qkv.out_features // 3, bias=to_qkv.bias is not None)
#         self.to_qk.weight.data.copy_(to_qkv.weight.data[0: to_qkv.out_features // 3 * 2])
#         if to_qkv.bias is not None:
#             self.to_qk.bias.data.copy_(to_qkv.bias.data[0: to_qkv.out_features // 3 * 2])
#         self.to_v.weight.data.copy_(to_qkv.weight.data[to_qkv.out_features // 3 * 2: ])
#         if to_qkv.bias is not None:
#             self.to_v.bias.data.copy_(to_qkv.bias.data[to_qkv.out_features // 3 * 2: ])
                
#         self.fbs = nn.Sequential(
#             Rearrange('b n d -> b d n'),
#             Abs(),
#             nn.AdaptiveAvgPool1d(1),
#             SqueezeLast(),
#             nn.Linear(to_qkv.in_features, to_qkv.out_features // 3 // r),
#             nn.ReLU(),
#             # nn.Linear(to_qkv.out_features // 3 // r, to_qkv.out_features // 3),
#             nn.Linear(to_qkv.out_features // 3 // r, self.to_v.out_features),
#             nn.ReLU()
#         )
        
#         nn.init.constant_(self.fbs[6].bias, 1.)
#         nn.init.kaiming_normal_(self.fbs[6].weight)
    
#     def forward(self, x):
#         if self.use_cached_channel_attention and self.cached_channel_attention is not None:
#             channel_attention = self.cached_channel_attention
#         else:
#             self.cached_raw_channel_attention = self.fbs(x)
            
#             # print()
#             # for attn in self.cached_raw_channel_attention.chunk(3, dim=1)[0: 1]:
#             #     print(self.cached_raw_channel_attention.size(), attn.size())
#             #     print(self.k_takes_all.k)
#             #     print(attn[0].nonzero(as_tuple=True)[0].size(), attn[0])
                
#             self.cached_channel_attention = self.k_takes_all(self.cached_raw_channel_attention)
            
            
#             # for attn in self.cached_channel_attention.chunk(3, dim=1)[0: 1]:
#             #     print(self.cached_channel_attention.size(), attn.size())
#             #     print(self.k_takes_all.k)
#             #     print(attn[0].nonzero(as_tuple=True)[0].size(), attn[0])
#             # print()
            
#             channel_attention = self.cached_channel_attention
        
#         qk = self.to_qk(x)
#         v = channel_attention.unsqueeze(1) * self.to_v(x)
#         return torch.cat([qk, v], dim=-1)
        
        # qkv = raw_res.chunk(3, dim = -1)
        
        # raw_v = qkv[2]
        # print('raw_k, raw_v', qkv[0].sum((0, 1))[0: 10], qkv[0].sum((0, 1)).nonzero(as_tuple=True)[0].size(),
        #       qkv[1].sum((0, 1))[0: 10], qkv[1].sum((0, 1)).nonzero(as_tuple=True)[0].size(),)
        # print('raw_v', raw_v.size(), raw_v.sum((0, 1))[0: 10], raw_v.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        
        # qkv_attn = channel_attention.chunk(3, dim=-1)
        # print('attn', [attn[0][0: 10] for attn in qkv_attn])
        # print(channel_attention.unsqueeze(1).size(), raw_res.size())
        # print('fbs', channel_attention.size(), raw_res.size())
        # return channel_attention.unsqueeze(1) * raw_res
    
    
class LinearStaticFBS(nn.Module):
    def __init__(self, static_channel_attention):
        super(LinearStaticFBS, self).__init__()
        assert static_channel_attention.dim() == 2 and static_channel_attention.size(0) == 1
        self.static_channel_attention = nn.Parameter(static_channel_attention, requires_grad=False) # (1, dim)
        
    def forward(self, x):
        # print('staticfbs', x, self.static_channel_attention.unsqueeze(1))
        return x * self.static_channel_attention.unsqueeze(1)
    
from .cnn import StaticFBS as ConvStaticFBS
    
    
class ElasticViTUtil(ElasticDNNUtil):
    def convert_raw_dnn_to_master_dnn(self, raw_dnn: nn.Module, r: float, ignore_layers=[]):
        assert len(ignore_layers) == 0, 'not supported yet'

        raw_vit = deepcopy(raw_dnn)
        
        set_module(raw_vit, 'patch_embed.proj', ProjConv_WrappedWithFBS(raw_vit.patch_embed.proj, r))
                
        for name, module in raw_vit.named_modules():
            if name.endswith('mlp'):
                set_module(module, 'fc1', Linear_WrappedWithFBS(module.fc1, r))
        
        return raw_vit
    
    # def set_master_dnn_sparsity(self, master_dnn: nn.Module, sparsity: float):
    #     for name, module in master_dnn.named_modules():
    #         if not name.endswith('attn'):
    #             continue
            
    #         q_features = module.qkv.to_qk.out_features // 2
            
    #         if (q_features - int(q_features * sparsity)) % module.num_heads != 0:
    #             # tune sparsity to ensure #unpruned channel % num_heads == 0
    #             # so that the pruning seems to reduce the dim_head of each head
    #             tuned_sparsity = 1. - int((q_features - int(q_features * sparsity)) / module.num_heads) * module.num_heads / q_features
    #             logger.debug(f'tune sparsity from {sparsity:.2f} to {tuned_sparsity}')
    #             sparsity = tuned_sparsity
    #             break
        
    #     return super().set_master_dnn_sparsity(master_dnn, sparsity)
    
    def select_most_rep_sample(self, master_dnn: nn.Module, samples: torch.Tensor):
        return samples[0].unsqueeze(0)
    
    def extract_surrogate_dnn_via_samples(self, master_dnn: nn.Module, samples: torch.Tensor):
        sample = self.select_most_rep_sample(master_dnn, samples)
        assert sample.dim() == 4 and sample.size(0) == 1
        
        
        print('WARN: for debug, modify cls_token and pos_embed')
        master_dnn.pos_embed.data = torch.zeros_like(master_dnn.pos_embed.data)
        
        print('before')
        master_dnn.eval()
        self.clear_cached_channel_attention_in_master_dnn(master_dnn)
        
        # debug: add hooks
        
        hooks = {
            'blocks_input': LayerActivation(master_dnn.blocks, True, 'cuda')
        }
        
        with torch.no_grad():
            master_dnn_output = master_dnn(sample)
        
        for k, v in hooks.items():
            print(f'{k}: {v.input.size()}')
        
        print('after')
        
        boosted_vit = master_dnn
        
        def get_unpruned_indexes_from_channel_attn(channel_attn: torch.Tensor, k):
            assert channel_attn.size(0) == 1, 'use A representative sample to generate channel attentions'
            res = channel_attn[0].nonzero(as_tuple=True)[0] 
            return res
        
        proj = boosted_vit.patch_embed.proj
        proj_unpruned_indexes = get_unpruned_indexes_from_channel_attn(
            proj.cached_channel_attention, proj.k_takes_all.k)
        
        # 1.1 prune proj itself
        proj_conv = proj.raw_conv2d
        new_proj = nn.Conv2d(proj_conv.in_channels, proj_unpruned_indexes.size(0), proj_conv.kernel_size, proj_conv.stride, proj_conv.padding,
                             proj_conv.dilation, proj_conv.groups, proj_conv.bias is not None, proj_conv.padding_mode, proj_conv.weight.device)
        new_proj.weight.data.copy_(proj_conv.weight.data[proj_unpruned_indexes])
        if new_proj.bias is not None:
            new_proj.bias.data.copy_(proj_conv.bias.data[proj_unpruned_indexes])
        set_module(boosted_vit.patch_embed, 'proj', nn.Sequential(new_proj, ConvStaticFBS(proj.cached_channel_attention[0][proj_unpruned_indexes])))
        
        # print(boosted_vit.pos_embed.size(), boosted_vit.cls_token.size())
        boosted_vit.pos_embed.data = boosted_vit.pos_embed.data[:, :, proj_unpruned_indexes]
        boosted_vit.cls_token.data = boosted_vit.cls_token.data[:, :, proj_unpruned_indexes]
            
        def reduce_linear_output(raw_linear: nn.Linear, layer_name, unpruned_indexes: torch.Tensor):
            new_linear = nn.Linear(raw_linear.in_features, unpruned_indexes.size(0), raw_linear.bias is not None)
            new_linear.weight.data.copy_(raw_linear.weight.data[unpruned_indexes])
            if raw_linear.bias is not None:
                new_linear.bias.data.copy_(raw_linear.bias.data[unpruned_indexes])
            set_module(boosted_vit, layer_name, new_linear)
            
        def reduce_linear_input(raw_linear: nn.Linear, layer_name, unpruned_indexes: torch.Tensor):
            new_linear = nn.Linear(unpruned_indexes.size(0), raw_linear.out_features, raw_linear.bias is not None)
            new_linear.weight.data.copy_(raw_linear.weight.data[:, unpruned_indexes])
            if raw_linear.bias is not None:
                new_linear.bias.data.copy_(raw_linear.bias.data)
            set_module(boosted_vit, layer_name, new_linear)
            
        def reduce_norm(raw_norm: nn.LayerNorm, layer_name, unpruned_indexes: torch.Tensor):
            new_norm = nn.LayerNorm(unpruned_indexes.size(0), raw_norm.eps, raw_norm.elementwise_affine)
            new_norm.weight.data.copy_(raw_norm.weight.data[unpruned_indexes])
            new_norm.bias.data.copy_(raw_norm.bias.data[unpruned_indexes])
            set_module(boosted_vit, layer_name, new_norm)
        
        # 1.2 prune blocks.x.norm1/to_qkv/proj/fc1/fc2
        for block_i, block in enumerate(boosted_vit.blocks):
            attn = block.attn
            ff = block.mlp
            
            reduce_norm(block.norm1, f'blocks.{block_i}.norm1', proj_unpruned_indexes)
            reduce_linear_input(attn.qkv, f'blocks.{block_i}.attn.qkv', proj_unpruned_indexes)
            reduce_linear_output(attn.proj, f'blocks.{block_i}.attn.proj', proj_unpruned_indexes)
            reduce_norm(block.norm2, f'blocks.{block_i}.norm2', proj_unpruned_indexes)
            reduce_linear_input(ff.fc1.linear, f'blocks.{block_i}.mlp.fc1.linear', proj_unpruned_indexes)
            reduce_linear_output(ff.fc2, f'blocks.{block_i}.mlp.fc2', proj_unpruned_indexes)
            
        # 1.3 prune norm, head
        reduce_norm(boosted_vit.norm, f'norm', proj_unpruned_indexes)
        reduce_linear_input(boosted_vit.head, f'head', proj_unpruned_indexes)
            
        # 2. prune blocks.x.fc1
        for block_i, block in enumerate(boosted_vit.blocks):
            attn = block.attn
            ff = block.mlp
            
            fc1 = ff.fc1
            fc1_unpruned_indexes = get_unpruned_indexes_from_channel_attn(fc1.cached_channel_attention, fc1.k_takes_all.k)
            fc1_linear = fc1.linear
            new_linear = nn.Linear(fc1_linear.in_features, fc1_unpruned_indexes.size(0), fc1_linear.bias is not None)
            new_linear.weight.data.copy_(fc1_linear.weight.data[fc1_unpruned_indexes])
            if fc1_linear.bias is not None:
                new_linear.bias.data.copy_(fc1_linear.bias.data[fc1_unpruned_indexes])
            set_module(boosted_vit, f'blocks.{block_i}.mlp.fc1', nn.Sequential(new_linear, LinearStaticFBS(fc1.cached_channel_attention[:, fc1_unpruned_indexes])))
            
            reduce_linear_input(ff.fc2, f'blocks.{block_i}.mlp.fc2', fc1_unpruned_indexes)
        
        
        
        surrogate_dnn = boosted_vit
        surrogate_dnn.eval()
        surrogate_dnn = surrogate_dnn.to(get_model_device(master_dnn))
        print(surrogate_dnn)
        
        
        hooks = {
            'blocks_input': LayerActivation(surrogate_dnn.blocks, True, 'cuda')
        }
        
        with torch.no_grad():
            surrogate_dnn_output = surrogate_dnn(sample)
        
        for k, v in hooks.items():
            print(f'{k}: {v.input.size()}')
        
        output_diff = ((surrogate_dnn_output - master_dnn_output) ** 2).sum()
        assert output_diff < 1e-4, output_diff
        logger.info(f'output diff of master and surrogate DNN: {output_diff}')
        
        return boosted_vit
    