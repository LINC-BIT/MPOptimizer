from copy import deepcopy
from typing import Optional, Union
import torch
from torch import nn 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import tqdm

from utils.dl.common.model import get_model_device, get_model_size, set_module, get_module
from .base import Abs, KTakesAll, ElasticDNNUtil, Layer_WrappedWithFBS
from utils.common.log import logger


class SqueezeLast(nn.Module):
    def __init__(self):
        super(SqueezeLast, self).__init__()
    
    def forward(self, x):
        return x.squeeze(-1)
    
    
class ProjConv_WrappedWithFBS(Layer_WrappedWithFBS):
    def __init__(self, proj: nn.Conv2d, r):
        super(ProjConv_WrappedWithFBS, self).__init__()
        
        self.proj = proj
        
        # for conv: (B, C_in, H, W) -> (B, C_in) -> (B, C_out)
        # for mlp in ViT: (B, #patches, D: dim of patches embedding) -> (B, D) -> (B, C_out)
        self.fbs = nn.Sequential(
            Abs(),
            nn.AdaptiveAvgPool1d(1),
            SqueezeLast(),
            nn.Linear(proj.in_channels, proj.out_channels // r),
            nn.ReLU(),
            nn.Linear(proj.out_channels // r, proj.out_channels),
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
        
        raw_res = self.proj(x)
        
        return channel_attention.unsqueeze(1) * raw_res # TODO:


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
    
    
class ToQKV_WrappedWithFBS(Layer_WrappedWithFBS):
    """
    This regards to_q/to_k/to_v as a whole (in fact it consists of multiple heads) and prunes it.
    It seems different channels of different heads are pruned according to the input. 
    This is different from "removing some head" or "removing the same channels in each head".
    """
    def __init__(self, to_qkv: nn.Linear, r):
        super(ToQKV_WrappedWithFBS, self).__init__()
        
        # self.to_qkv = to_qkv
        
        self.to_qk = nn.Linear(to_qkv.in_features, to_qkv.out_features // 3 * 2, bias=to_qkv.bias is not None)
        self.to_v = nn.Linear(to_qkv.in_features, to_qkv.out_features // 3, bias=to_qkv.bias is not None)
        self.to_qk.weight.data.copy_(to_qkv.weight.data[0: to_qkv.out_features // 3 * 2])
        if to_qkv.bias is not None:
            self.to_qk.bias.data.copy_(to_qkv.bias.data[0: to_qkv.out_features // 3 * 2])
        self.to_v.weight.data.copy_(to_qkv.weight.data[to_qkv.out_features // 3 * 2: ])
        if to_qkv.bias is not None:
            self.to_v.bias.data.copy_(to_qkv.bias.data[to_qkv.out_features // 3 * 2: ])
                
        self.fbs = nn.Sequential(
            Rearrange('b n d -> b d n'),
            Abs(),
            nn.AdaptiveAvgPool1d(1),
            SqueezeLast(),
            nn.Linear(to_qkv.in_features, to_qkv.out_features // 3 // r),
            nn.ReLU(),
            # nn.Linear(to_qkv.out_features // 3 // r, to_qkv.out_features // 3),
            nn.Linear(to_qkv.out_features // 3 // r, self.to_v.out_features),
            nn.ReLU()
        )
        
        nn.init.constant_(self.fbs[6].bias, 1.)
        nn.init.kaiming_normal_(self.fbs[6].weight)
    
    def forward(self, x):
        if self.use_cached_channel_attention and self.cached_channel_attention is not None:
            channel_attention = self.cached_channel_attention
        else:
            self.cached_raw_channel_attention = self.fbs(x)
            
            # print()
            # for attn in self.cached_raw_channel_attention.chunk(3, dim=1)[0: 1]:
            #     print(self.cached_raw_channel_attention.size(), attn.size())
            #     print(self.k_takes_all.k)
            #     print(attn[0].nonzero(as_tuple=True)[0].size(), attn[0])
                
            self.cached_channel_attention = self.k_takes_all(self.cached_raw_channel_attention)
            
            
            # for attn in self.cached_channel_attention.chunk(3, dim=1)[0: 1]:
            #     print(self.cached_channel_attention.size(), attn.size())
            #     print(self.k_takes_all.k)
            #     print(attn[0].nonzero(as_tuple=True)[0].size(), attn[0])
            # print()
            
            channel_attention = self.cached_channel_attention
        
        qk = self.to_qk(x)
        v = channel_attention.unsqueeze(1) * self.to_v(x)
        return torch.cat([qk, v], dim=-1)
        
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
    
    
class StaticFBS(nn.Module):
    def __init__(self, static_channel_attention):
        super(StaticFBS, self).__init__()
        assert static_channel_attention.dim() == 2 and static_channel_attention.size(0) == 1
        self.static_channel_attention = nn.Parameter(static_channel_attention, requires_grad=False) # (1, dim)
        
    def forward(self, x):
        # print('staticfbs', x, self.static_channel_attention.unsqueeze(1))
        return x * self.static_channel_attention.unsqueeze(1)
    
    
class ElasticBertUtil(ElasticDNNUtil):
    def convert_raw_dnn_to_master_dnn(self, raw_dnn: nn.Module, r: float, ignore_layers=[]):
        assert len(ignore_layers) == 0, 'not supported yet'

        raw_vit = deepcopy(raw_dnn)
        
        # set_module(module, 'patch_embed.proj', ProjConv_WrappedWithFBS(module.patch_embed.proj, r))
                
        for name, module in raw_vit.named_modules():
            # if name.endswith('attn'):
            #     set_module(module, 'qkv', ToQKV_WrappedWithFBS(module.qkv, r))
            if name.endswith('intermediate'):
                set_module(module, 'dense', Linear_WrappedWithFBS(module.dense, r))
        
        return raw_vit
    
    def set_master_dnn_sparsity(self, master_dnn: nn.Module, sparsity: float):
        # for name, module in master_dnn.named_modules():
        #     if not name.endswith('attn'):
        #         continue
            
        #     q_features = module.qkv.to_qk.out_features // 2
            
        #     if (q_features - int(q_features * sparsity)) % module.num_heads != 0:
        #         # tune sparsity to ensure #unpruned channel % num_heads == 0
        #         # so that the pruning seems to reduce the dim_head of each head
        #         tuned_sparsity = 1. - int((q_features - int(q_features * sparsity)) / module.num_heads) * module.num_heads / q_features
        #         logger.debug(f'tune sparsity from {sparsity:.2f} to {tuned_sparsity}')
        #         sparsity = tuned_sparsity
        #         break
        
        return super().set_master_dnn_sparsity(master_dnn, sparsity)
    
    def select_most_rep_sample(self, master_dnn: nn.Module, samples: torch.Tensor):
        # print(samples)
        # return samples[0].unsqueeze(0)
        res = {k: v[0: 1] for k, v in samples.items()}
        return res
        
    def extract_surrogate_dnn_via_samples(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):
        sample = self.select_most_rep_sample(master_dnn, samples)
        # assert sample.dim() == 4 and sample.size(0) == 1
        
        # print('before')
        master_dnn.eval()
        self.clear_cached_channel_attention_in_master_dnn(master_dnn)
        with torch.no_grad():
            master_dnn_output = master_dnn(**sample)
            
        # print('after')
        
        boosted_vit = deepcopy(master_dnn)
        
        def get_unpruned_indexes_from_channel_attn(channel_attn: torch.Tensor, k):
            assert channel_attn.size(0) == 1, 'use A representative sample to generate channel attentions'
            
            # print('attn_in_unpruned', channel_attn[0][0: 10])
            
            res = channel_attn[0].nonzero(as_tuple=True)[0] # should be one-dim

            # res = channel_attn[0].argsort(descending=True)[0: -int(channel_attn.size(1) * k)].sort()[0]
            
            # g = channel_attn
            # k = g.size(1) - int(g.size(1) * k)
            # res = g.topk(k, 1)[1][0].sort()[0]
            
            return res
        
        unpruned_indexes_of_layers = {}
        
        # for attn, ff in boosted_vit.transformer.layers:
        # for block_i, block in enumerate(boosted_vit.blocks):
        for block_i, block in enumerate(boosted_vit.bert.encoder.layer):
            # attn = block.attn
            # ff = block.mlp
            
            ff_0 = get_module(block, f'intermediate.dense')
            # ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, k)
            ff_0_pruned_indexes = ff_0.k_takes_all.cached_i[0].sort()[0]
            ff_0_unpruned_indexes = torch.LongTensor([ii for ii in range(ff_0.cached_channel_attention.size(1)) if ii not in ff_0_pruned_indexes])
            new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
            new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
            if ff_0.linear.bias is not None:
                new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
            set_module(block, 'intermediate.dense', nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
            
            ff_1 = get_module(block, f'output.dense')
            new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
            new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
            if ff_1.bias is not None:
                new_ff_1.bias.data.copy_(ff_1.bias.data)
            set_module(block, 'output.dense', new_ff_1)
            
            unpruned_indexes_of_layers[f'bert.encoder.layer.{block_i}.intermediate.dense.0.weight'] = ff_0_unpruned_indexes
        
        surrogate_dnn = boosted_vit
        surrogate_dnn.eval()
        surrogate_dnn = surrogate_dnn.to(get_model_device(master_dnn))
        # logger.debug(surrogate_dnn)
        with torch.no_grad():
            surrogate_dnn_output = surrogate_dnn(**sample)
            
        output_diff = ((surrogate_dnn_output - master_dnn_output) ** 2).sum()
        # assert output_diff < 1e-4, output_diff
        logger.info(f'output diff of master and surrogate DNN: {output_diff}')
        logger.debug(f'example output of master/surrogate: {master_dnn_output.sum(0)[0: 10]}, {surrogate_dnn_output.sum(0)[0: 10]}')
        # logger.info(f'\nonly prune mlp!!!!\n')
        # logger.info(f'\nonly prune mlp!!!!\n')
        
        if return_detail:
            return boosted_vit, unpruned_indexes_of_layers
        
        return boosted_vit
    
    def extract_surrogate_dnn_via_samples_with_perf_test(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):
        master_dnn_size = get_model_size(master_dnn, True)
        master_dnn_latency = self._get_model_latency(master_dnn, samples, 50, 
                                               get_model_device(master_dnn), 50, False)
        
        res = self.extract_surrogate_dnn_via_samples(master_dnn, samples, return_detail)
        if not return_detail:
            surrogate_dnn = res
        else:
            surrogate_dnn, unpruned_indexes_of_layers = res
        surrogate_dnn_size = get_model_size(surrogate_dnn, True)
        surrogate_dnn_latency = self._get_model_latency(master_dnn, samples, 50, 
                                               get_model_device(master_dnn), 50, False)

        logger.info(f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample) -> '
                    f'surrogate DNN ({surrogate_dnn_size:.3f}MB, {surrogate_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(master_dnn_size / surrogate_dnn_size):.2f}x, '
                    f'latency: ↓ {(master_dnn_latency / surrogate_dnn_latency):.2f}x)')
        
        return res
    
    def _get_model_latency(self, model: torch.nn.Module, model_input_size, sample_num: int, 
                           device: str, warmup_sample_num: int, return_detail=False):
        import time
        
        if isinstance(model_input_size, tuple):
            dummy_input = torch.rand(model_input_size).to(device)
        else:
            dummy_input = model_input_size
            
        model = model.to(device)
        model.eval()
        
        # warm up
        with torch.no_grad():
            for _ in range(warmup_sample_num):
                model(**dummy_input)
                
        infer_time_list = []
                
        if device == 'cuda' or 'cuda' in str(device):
            with torch.no_grad():
                for _ in range(sample_num):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    model(**dummy_input)
                    e.record()
                    torch.cuda.synchronize()
                    cur_model_infer_time = s.elapsed_time(e) / 1000.
                    infer_time_list += [cur_model_infer_time]

        else:
            with torch.no_grad():
                for _ in range(sample_num):
                    start = time.time()
                    model(**dummy_input)
                    cur_model_infer_time = time.time() - start
                    infer_time_list += [cur_model_infer_time]
                    
        avg_infer_time = sum(infer_time_list) / sample_num

        if return_detail:
            return avg_infer_time, infer_time_list
        return avg_infer_time