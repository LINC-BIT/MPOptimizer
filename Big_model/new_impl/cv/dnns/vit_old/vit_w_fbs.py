from copy import deepcopy
from typing import Optional, Union
import torch
from torch import nn 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from raw_vit import ViT, Attention, FeedForward
from utils.dl.common.model import get_model_size, set_module


class KTakesAll(nn.Module):
    # k means sparsity (the larger k is, the smaller model is)
    def __init__(self, k):
        super(KTakesAll, self).__init__()
        self.k = k
        
    def forward(self, g: torch.Tensor):
        k = int(g.size(1) * self.k)
        
        i = (-g).topk(k, 1)[1]
        t = g.scatter(1, i, 0)
                
        return t


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()
        
    def forward(self, x):
        return x.abs()


class SqueezeLast(nn.Module):
    def __init__(self):
        super(SqueezeLast, self).__init__()
    
    def forward(self, x):
        return x.squeeze(-1)


class Linear_WrappedWithFBS(nn.Module):
    def __init__(self, linear: nn.Linear, r, k):
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
            nn.ReLU(),
            KTakesAll(k)
        )
        self.k = k
        
        self.cached_channel_attention = None # (batch_size, dim)
        self.use_cached_channel_attention = False
    
    def forward(self, x):
        if self.use_cached_channel_attention and self.cached_channel_attention is not None:
            channel_attention = self.cached_channel_attention
        else:
            channel_attention = self.fbs(x)
            self.cached_channel_attention = channel_attention
        
        raw_res = self.linear(x)
        return channel_attention.unsqueeze(1) * raw_res
    
    
class ToQKV_WrappedWithFBS(nn.Module):
    """
    This regards to_q/to_k/to_v as a whole (in fact it consists of multiple heads) and prunes it.
    It seems different channels of different heads are pruned according to the input. 
    This is different from "removing some head" or "removing the same channels in each head".
    """
    def __init__(self, to_qkv: nn.Linear, r, k):
        super(ToQKV_WrappedWithFBS, self).__init__()
        
        self.to_qkv = to_qkv
        self.fbses = nn.ModuleList([nn.Sequential(
            Rearrange('b n d -> b d n'),
            Abs(),
            nn.AdaptiveAvgPool1d(1),
            SqueezeLast(),
            nn.Linear(to_qkv.in_features, to_qkv.out_features // 3 // r),
            nn.ReLU(),
            nn.Linear(to_qkv.out_features // 3 // r, to_qkv.out_features // 3),
            nn.ReLU(),
            KTakesAll(k)
        ) for _ in range(3)])
        self.k = k
        
        self.cached_channel_attention = None
        self.use_cached_channel_attention = False
    
    def forward(self, x):
        if self.use_cached_channel_attention and self.cached_channel_attention is not None:
            # print('use cache')
            channel_attention = self.cached_channel_attention
        else:
            # print('dynamic')
            channel_attention = torch.cat([fbs(x) for fbs in self.fbses], dim=1)
            self.cached_channel_attention = channel_attention
        
        raw_res = self.to_qkv(x)
        return channel_attention.unsqueeze(1) * raw_res
        
        
def boost_raw_vit_by_fbs(raw_vit: ViT, r, k):
    raw_vit = deepcopy(raw_vit)
    
    raw_vit_model_size = get_model_size(raw_vit, True)
    
    # set_module(raw_vit.to_patch_embedding, '2', Linear_WrappedWithFBS(raw_vit.to_patch_embedding[2], r, k))
    
    for attn, ff in raw_vit.transformer.layers:
        attn = attn.fn
        ff = ff.fn
        
        set_module(attn, 'to_qkv', ToQKV_WrappedWithFBS(attn.to_qkv, r, k))
        set_module(ff.net, '0', Linear_WrappedWithFBS(ff.net[0], r, k))
        
    boosted_vit_model_size = get_model_size(raw_vit, True)
    
    print(f'boost_raw_vit_by_fbs() | model size from {raw_vit_model_size:.3f}MB to {boosted_vit_model_size:.3f}MB '
          f'(↑ {((boosted_vit_model_size - raw_vit_model_size) / raw_vit_model_size * 100):.2f}%)')
        
    return raw_vit


def set_boosted_vit_sparsity(boosted_vit: ViT, sparsity: float):
    for attn, ff in boosted_vit.transformer.layers:
        attn = attn.fn
        ff = ff.fn
        
        q_features = attn.to_qkv.to_qkv.out_features // 3
        
        if (q_features - int(q_features * sparsity)) % attn.heads != 0:
            # tune sparsity to ensure #unpruned channel % num_heads == 0
            # so that the pruning seems to reduce the dim_head of each head
            tuned_sparsity = 1. - int((q_features - int(q_features * sparsity)) / attn.heads) * attn.heads / q_features
            print(f'set_boosted_vit_sparsity() | tune sparsity from {sparsity} to {tuned_sparsity}')
            sparsity = tuned_sparsity
        
        attn.to_qkv.k = sparsity
        for fbs in attn.to_qkv.fbses:
            fbs[-1].k = sparsity
        ff.net[0].k = sparsity
        ff.net[0].fbs[-1].k = sparsity


def set_boosted_vit_inference_via_cached_channel_attentions(boosted_vit: ViT):
    for attn, ff in boosted_vit.transformer.layers:
        attn = attn.fn
        ff = ff.fn
        
        assert attn.to_qkv.cached_channel_attention is not None
        assert ff.net[0].cached_channel_attention is not None
        
        attn.to_qkv.use_cached_channel_attention = True
        ff.net[0].use_cached_channel_attention = True
        
        
def set_boosted_vit_dynamic_inference(boosted_vit: ViT):
    for attn, ff in boosted_vit.transformer.layers:
        attn = attn.fn
        ff = ff.fn
        
        attn.to_qkv.use_cached_channel_attention = False
        ff.net[0].use_cached_channel_attention = False
        
        
class StaticFBS(nn.Module):
    def __init__(self, static_channel_attention):
        super(StaticFBS, self).__init__()
        assert static_channel_attention.dim() == 2 and static_channel_attention.size(0) == 1
        self.static_channel_attention = nn.Parameter(static_channel_attention, requires_grad=False) # (1, dim)
        
    def forward(self, x):
        return x * self.static_channel_attention.unsqueeze(1)


def extract_surrogate_vit_via_cached_channel_attn(boosted_vit: ViT):
    boosted_vit = deepcopy(boosted_vit)
    raw_vit_model_size = get_model_size(boosted_vit, True)
    
    def get_unpruned_indexes_from_channel_attn(channel_attn: torch.Tensor, k):
        assert channel_attn.size(0) == 1, 'use A representative sample to generate channel attentions'
        
        res = channel_attn[0].nonzero(as_tuple=True)[0] # should be one-dim
        return res
    
    for attn, ff in boosted_vit.transformer.layers:
        attn = attn.fn
        ff_w_norm = ff
        ff = ff_w_norm.fn
        
        # prune to_qkv
        to_qkv = attn.to_qkv
        to_q_unpruned_indexes = get_unpruned_indexes_from_channel_attn(
            to_qkv.cached_channel_attention[:, 0: to_qkv.cached_channel_attention.size(1) // 3],
            to_qkv.k
        )
        to_q_unpruned_indexes_w_offset = to_q_unpruned_indexes
        to_k_unpruned_indexes = get_unpruned_indexes_from_channel_attn(
            to_qkv.cached_channel_attention[:, to_qkv.cached_channel_attention.size(1) // 3: to_qkv.cached_channel_attention.size(1) // 3 * 2],
            to_qkv.k
        )
        to_k_unpruned_indexes_w_offset = to_k_unpruned_indexes + to_qkv.cached_channel_attention.size(1) // 3
        to_v_unpruned_indexes = get_unpruned_indexes_from_channel_attn(
            to_qkv.cached_channel_attention[:, to_qkv.cached_channel_attention.size(1) // 3 * 2: ],
            to_qkv.k
        )
        to_v_unpruned_indexes_w_offset = to_v_unpruned_indexes + to_qkv.cached_channel_attention.size(1) // 3 * 2
        assert to_q_unpruned_indexes.size(0) == to_k_unpruned_indexes.size(0) == to_v_unpruned_indexes.size(0)
        to_qkv_unpruned_indexes = torch.cat([to_q_unpruned_indexes_w_offset, to_k_unpruned_indexes_w_offset, to_v_unpruned_indexes_w_offset])
        new_to_qkv = nn.Linear(to_qkv.to_qkv.in_features, to_qkv_unpruned_indexes.size(0), to_qkv.to_qkv.bias is not None)
        new_to_qkv.weight.data.copy_(to_qkv.to_qkv.weight.data[to_qkv_unpruned_indexes])
        if to_qkv.to_qkv.bias is not None:
            new_to_qkv.bias.data.copy_(to_qkv.to_qkv.bias.data[to_qkv_unpruned_indexes])
        set_module(attn, 'to_qkv', nn.Sequential(new_to_qkv, StaticFBS(to_qkv.cached_channel_attention[:, to_qkv_unpruned_indexes])))
        
        # prune to_out
        to_out = attn.to_out[0]
        new_to_out = nn.Linear(to_v_unpruned_indexes.size(0), to_out.out_features, to_out.bias is not None)
        new_to_out.weight.data.copy_(to_out.weight.data[:, to_v_unpruned_indexes])
        if to_out.bias is not None:
            new_to_out.bias.data.copy_(to_out.bias.data)
        set_module(attn, 'to_out', new_to_out)
        
        ff_0 = ff.net[0]
        ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, ff_0.k)
        new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
        new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
        if ff_0.linear.bias is not None:
            new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
        set_module(ff.net, '0', nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
        
        ff_1 = ff.net[3]
        new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
        new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
        if ff_1.bias is not None:
            new_ff_1.bias.data.copy_(ff_1.bias.data)
        set_module(ff.net, '3', new_ff_1)
        
    pruned_vit_model_size = get_model_size(boosted_vit, True)
    
    print(f'extract_surrogate_vit_via_cached_channel_attn() | model size from {raw_vit_model_size:.3f}MB to {pruned_vit_model_size:.3f}MB '
          f'({(pruned_vit_model_size / raw_vit_model_size * 100):.2f}%)')
        
    return boosted_vit
    
    
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    def verify(vit, sparsity=0.8):
        vit.eval()
        
        with torch.no_grad():
            r = torch.rand((1, 3, 224, 224))
            print(vit(r).size())
        # print(vit)
        
        boosted_vit = boost_raw_vit_by_fbs(vit, r=32, k=sparsity)
        set_boosted_vit_sparsity(boosted_vit, sparsity)
        # print(boosted_vit)
        with torch.no_grad():
            r = torch.rand((1, 3, 224, 224))
            print(boosted_vit(r).size())
            
        # set_boosted_vit_inference_via_cached_channel_attentions(boosted_vit)
        r = torch.rand((1, 3, 224, 224))
        boosted_vit.eval()
        with torch.no_grad():
            o1 = boosted_vit(r)
            
        pruned_vit = extract_surrogate_vit_via_cached_channel_attn(boosted_vit)
        pruned_vit.eval()
        with torch.no_grad():
            o2 = pruned_vit(r)
            print('output diff (should be tiny): ', ((o1 - o2) ** 2).sum())
            
        # print(pruned_vit)
        # print(pruned_vit)
    
    # vit_b_16 = ViT(
    #     image_size = 224,
    #     patch_size = 16,
    #     num_classes = 1000,
    #     dim = 768, # encoder layer/attention input/output size (Hidden Size D in the paper)
    #     depth = 12,
    #     heads = 12, # (Heads in the paper)
    #     dim_head = 64, # attention hidden size (seems be default, never change this)
    #     mlp_dim = 3072, # mlp layer hidden size (MLP size in the paper)
    #     dropout = 0.,
    #     emb_dropout = 0.
    # )
    # verify(vit_b_16)
    
    vit_l_16 = ViT(
        image_size = 224,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024, # encoder layer/attention input/output size (Hidden Size D in the paper)
        depth = 24,
        heads = 16, # (Heads in the paper)
        dim_head = 64, # attention hidden size (seems be default, never change this)
        mlp_dim = 4096, # mlp layer hidden size (MLP size in the paper)
        dropout = 0.,
        emb_dropout = 0.
    )
    verify(vit_l_16, 0.98)
    
    # vit_h_16 = ViT(
    #     image_size = 224,
    #     patch_size = 16,
    #     num_classes = 1000,
    #     dim = 1280, # encoder layer/attention input/output size (Hidden Size D in the paper)
    #     depth = 32,
    #     heads = 16, # (Heads in the paper)
    #     dim_head = 64, # attention hidden size (seems be default, never change this)
    #     mlp_dim = 5120, # mlp layer hidden size (MLP size in the paper)
    #     dropout = 0.,
    #     emb_dropout = 0.
    # )
    # verify(vit_h_16)