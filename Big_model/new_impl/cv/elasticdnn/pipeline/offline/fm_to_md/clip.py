
import torch
from torch import nn 
from copy import deepcopy

from .base import FM_to_MD_Util
from utils.common.log import logger
from utils.dl.common.model import set_module, get_module, get_super_module
from utils.dl.common.model import get_model_device, get_model_latency, get_model_size
from utils.common.log import logger
from typing import Optional, Tuple

from transformers.models.clip.modeling_clip import CLIPAttention
from transformers import CLIPVisionConfig


class CLIPAttentionPrunable(CLIPAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""


    def __init__(self):
        config = CLIPVisionConfig.from_pretrained('openai/clip-vit-base-patch16')
        super(CLIPAttentionPrunable, self).__init__(config)
    
    
    # def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    #     # print(tensor.size(), self.num_heads, self.head_dim, bsz) # torch.Size([1, 197, 192]) 8 64 1
    #     # head_dim should be modified
        
    #     # 'b n (h d) -> b h n d', h = self.num_heads
        
    #     if seq_len == -1:
    #         seq_len = tensor.size(1)
            
    #     # print(tensor.size(), bsz, seq_len, self.num_heads, -1)
    #     return tensor.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2).contiguous()

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     causal_attention_mask: Optional[torch.Tensor] = None,
    #     output_attentions: Optional[bool] = False,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     """Input shape: Batch x Time x Channel"""

    #     bsz, tgt_len, embed_dim = hidden_states.size()

    #     # get query proj
    #     query_states = self.q_proj(hidden_states) * self.scale
    #     key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    #     value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    #     proj_shape = (-1, tgt_len, self.head_dim)
    #     # print(proj_shape, key_states.size())
    #     query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    #     key_states = key_states.view(*proj_shape)
    #     value_states = value_states.view(*proj_shape)

    #     src_len = key_states.size(1)
    #     attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    #     # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    #     #     raise ValueError(
    #     #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
    #     #         f" {attn_weights.size()}"
    #     #     )

    #     # apply the causal_attention_mask first
    #     if causal_attention_mask is not None:
    #         if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
    #             raise ValueError(
    #                 f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
    #                 f" {causal_attention_mask.size()}"
    #             )
    #         attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
    #         attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    #     if attention_mask is not None:
    #         if attention_mask.size() != (bsz, 1, tgt_len, src_len):
    #             raise ValueError(
    #                 f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
    #             )
    #         attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    #         attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    #     if output_attentions:
    #         # this operation is a bit akward, but it's required to
    #         # make sure that attn_weights keeps its gradient.
    #         # In order to do so, attn_weights have to reshaped
    #         # twice and have to be reused in the following
    #         attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
    #         attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    #     else:
    #         attn_weights_reshaped = None

    #     attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    #     attn_output = torch.bmm(attn_probs, value_states)

    #     # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    #     #     raise ValueError(
    #     #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
    #     #         f" {attn_output.size()}"
    #     #     )

    #     attn_output = attn_output.view(bsz, self.num_heads, tgt_len, -1)
    #     attn_output = attn_output.transpose(1, 2)
    #     attn_output = attn_output.reshape(bsz, tgt_len, -1)

    #     attn_output = self.out_proj(attn_output)

    #     return attn_output, attn_weights_reshaped
    
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def _shape_dynamic_head_dim(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2).contiguous()

    def _shape_dynamic_num_head(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # logger.info(f'hidden state size: {hidden_states.size()}') # (64, 197, 768)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape_dynamic_head_dim(self.k_proj(hidden_states), tgt_len, bsz)
        value_states = self._shape_dynamic_head_dim(self.v_proj(hidden_states), tgt_len, bsz)
        
        # (64, 197, 768), numhead: 12, head_dim: 64, seq_len: 197
        # logger.info(f'key states: {self.k_proj(hidden_states).size()}, bsz: {bsz}, num_heads: {self.num_heads}, head_dim: {self.head_dim}, '
        #             f'seq_len: {self.k_proj(hidden_states).numel() / bsz / self.num_heads / self.head_dim}')
        # (64, 197, 768), numhead: 12, head_dim: 64, seq_len: 197
        # logger.info(f'value states: {self.v_proj(hidden_states).size()}, bsz: {bsz}, num_heads: {self.num_heads}, head_dim: {self.head_dim}, '
                    # f'seq_len: {self.v_proj(hidden_states).numel() / bsz / self.num_heads / self.head_dim}')

        proj_shape = (bsz * self.num_heads, tgt_len, -1)
        query_states = self._shape_dynamic_head_dim(query_states, tgt_len, bsz).view(*proj_shape)
        
        # (64, 12, 197, 64), -1 means 197
        # logger.info(f'query states: {self._shape(query_states, tgt_len, bsz).size()}, '
        #             f'-1 in proj_shape: {self._shape(query_states, tgt_len, bsz).numel() / bsz / self.num_heads / self.head_dim}')
        
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )
        # print(attn_output.size(), bsz, tgt_len, embed_dim)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, -1)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, -1)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped
    
    # reduce num_head
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     causal_attention_mask: Optional[torch.Tensor] = None,
    #     output_attentions: Optional[bool] = False,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     """Input shape: Batch x Time x Channel"""

    #     bsz, tgt_len, embed_dim = hidden_states.size()

    #     # logger.info(f'hidden state size: {hidden_states.size()}') # (64, 197, 768)

    #     # get query proj
    #     query_states = self.q_proj(hidden_states) * self.scale
    #     key_states = self._shape_dynamic_num_head(self.k_proj(hidden_states), tgt_len, bsz)
    #     value_states = self._shape_dynamic_num_head(self.v_proj(hidden_states), tgt_len, bsz)
        
    #     # (64, 197, 768), numhead: 12, head_dim: 64, seq_len: 197
    #     # logger.info(f'key states: {self.k_proj(hidden_states).size()}, bsz: {bsz}, num_heads: {self.num_heads}, head_dim: {self.head_dim}, '
    #     #             f'seq_len: {self.k_proj(hidden_states).numel() / bsz / self.num_heads / self.head_dim}')
    #     # (64, 197, 768), numhead: 12, head_dim: 64, seq_len: 197
    #     # logger.info(f'value states: {self.v_proj(hidden_states).size()}, bsz: {bsz}, num_heads: {self.num_heads}, head_dim: {self.head_dim}, '
    #                 # f'seq_len: {self.v_proj(hidden_states).numel() / bsz / self.num_heads / self.head_dim}')

    #     proj_shape = (-1, tgt_len, self.head_dim)
    #     query_states = self._shape_dynamic_head_dim(query_states, tgt_len, bsz).view(*proj_shape)
        
    #     # (64, 12, 197, 64), -1 means 197
    #     # logger.info(f'query states: {self._shape(query_states, tgt_len, bsz).size()}, '
    #     #             f'-1 in proj_shape: {self._shape(query_states, tgt_len, bsz).numel() / bsz / self.num_heads / self.head_dim}')
        
    #     key_states = key_states.view(*proj_shape)
    #     value_states = value_states.view(*proj_shape)

    #     src_len = key_states.size(1)
    #     attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    #     # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    #     #     raise ValueError(
    #     #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
    #     #         f" {attn_weights.size()}"
    #     #     )

    #     # apply the causal_attention_mask first
    #     if causal_attention_mask is not None:
    #         if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
    #             raise ValueError(
    #                 f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
    #                 f" {causal_attention_mask.size()}"
    #             )
    #         attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
    #         attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    #     if attention_mask is not None:
    #         if attention_mask.size() != (bsz, 1, tgt_len, src_len):
    #             raise ValueError(
    #                 f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
    #             )
    #         attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    #         attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    #     if output_attentions:
    #         # this operation is a bit akward, but it's required to
    #         # make sure that attn_weights keeps its gradient.
    #         # In order to do so, attn_weights have to reshaped
    #         # twice and have to be reused in the following
    #         attn_weights_reshaped = attn_weights.view(bsz, -1, tgt_len, src_len)
    #         attn_weights = attn_weights_reshaped.view(-1, tgt_len, src_len)
    #     else:
    #         attn_weights_reshaped = None

    #     attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    #     attn_output = torch.bmm(attn_probs, value_states)

    #     # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    #     #     raise ValueError(
    #     #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
    #     #         f" {attn_output.size()}"
    #     #     )
    #     # print(attn_output.size(), bsz, tgt_len, embed_dim)
    #     attn_output = attn_output.view(bsz, -1, tgt_len, self.head_dim)
    #     attn_output = attn_output.transpose(1, 2)
    #     attn_output = attn_output.reshape(bsz, tgt_len, -1)

    #     attn_output = self.out_proj(attn_output)

    #     return attn_output, attn_weights_reshaped
    
    
    @staticmethod
    def init_from_exist_self_attn(attn: CLIPAttention):
        # print(attn)
        
        res = CLIPAttentionPrunable()
        
        for attr in dir(attn):
            # if str(attr) in ['transpose_for_scores'] or str(attr).startswith('_'):
            #     continue
            # if isinstance(getattr(attn, attr), nn.Module):
                # print(attr)
                
            if isinstance(getattr(attn, attr), nn.Module):
                try:
                    # print(attr, 'ok')
                    setattr(res, attr, getattr(attn, attr))
                    
                except Exception as e:
                    print(attr, str(e))
        
        
        
        return res
    

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PrunableAttention(nn.Module):
    """
    https://github.com/lucidrains/vit-pytorch
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qkv_bias = False):
        super().__init__()
        self.inner_dim = inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.num_heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim * 3, bias = qkv_bias)

        # self.proj = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()
        
        self.proj = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,):
        
        x = hidden_states
        assert attention_mask is None
        assert causal_attention_mask is None
        assert not output_attentions
        # qkv = self.qkv(x).chunk(3, dim = -1)
        raw_qkv = self.qkv(x)
        
        self.inner_dim = (raw_qkv.size(-1) - self.proj.in_features) // 2
        qkv = raw_qkv[:, :, 0: self.inner_dim], raw_qkv[:, :, self.inner_dim: self.inner_dim * 2], raw_qkv[:, :, self.inner_dim * 2:]
        
        # print('v', qkv[0].size(), qkv[0].sum((0, 1))[0: 10], qkv[0].sum((0, 1)).nonzero(as_tuple=True)[0].size())
        
        # raw_v = qkv[2]
        # print('after_fbs_q, after_fbs_k', qkv[0].sum((0, 1))[0: 10], qkv[0].sum((0, 1)).nonzero(as_tuple=True)[0].size(),
        #       qkv[1].sum((0, 1))[0: 10], qkv[1].sum((0, 1)).nonzero(as_tuple=True)[0].size(),)
        # print('after_fbs_v', raw_v.size(), raw_v.sum((0, 1))[0: 10], raw_v.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        # print('q, before rearrage', qkv[0].size())
        q, k, v = qkv
        # print('raw qkv size', q.size(), k.size(), v.size())
        # exit()
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        # print('raw qkv size', q.size(), k.size(), v.size())
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # print('q, k, dots, after rearrage', q.size(), k.transpose(-1, -2).size(), dots.size())
        
        attn = self.attend(dots)
        # attn = dots
        attn = self.dropout(attn)

        # print(attn)
        # print('attn', attn.size(), attn.sum((0, 1))[0: 10], attn.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        # print('attn', attn.size(), attn.sum((0, 1))[0: 10], attn.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        # print('v2', v.size())
        out = torch.matmul(attn, v)
        # print('out1', out.size())
        # NOTE: just for trial debug
        # out = v
        
        # print('out before rerange', out.size())
        
        # print(v.size(), v)
        # exit()
        
        out = rearrange(out, 'b h n d -> b n (h d)')

        # print('out', out.size(), out.sum((0, 1))[0: 10], out.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        # exit()
        
        res = self.proj_dropout(self.proj(out))
        
        # res = self.proj_dropout(
        #     F.linear(self.proj.weight.T, out.T, self.proj.bias)
        # )
        # print(self.proj, self.proj_dropout)
        # print('res', res.size(), res.sum((0, 1))[0: 10], res.sum((0, 1)).nonzero(as_tuple=True)[0].size())

        return res, None


class FM_to_MD_CLIP_Util(FM_to_MD_Util):
    def init_md_from_fm_by_reducing_width(self, fm: nn.Module, reducing_width_ratio: int) -> nn.Module:
        fm_vit = deepcopy(fm)
        
        
        # for block in fm_vit.model.text_model.encoder.layers:
        #     set_module(block, 'self_attn', CLIPAttentionPrunable.init_from_exist_self_attn(block.self_attn))
        
        debug_input = torch.rand((1, 3, 32, 32)).cuda()
        fm.eval()
        o1 = fm.model.vision_model(debug_input).pooler_output
        for block in fm_vit.model.vision_model.encoder.layers:
            # set_module(block, 'self_attn', CLIPAttentionPrunable.init_from_exist_self_attn(block.self_attn))
            
            attn: CLIPAttention = block.self_attn
            # from dnns.vit import PrunableAttention
            new_attn = PrunableAttention(
                dim=768,
                heads=12,
                dim_head=64,
                dropout=0,
                qkv_bias=True
            )
            new_attn.qkv.weight.data.copy_(torch.cat([
                attn.q_proj.weight,
                attn.k_proj.weight,
                attn.v_proj.weight
            ], dim=0))
            new_attn.qkv.bias.data.copy_(torch.cat([
                attn.q_proj.bias,
                attn.k_proj.bias,
                attn.v_proj.bias
            ], dim=0))
            new_attn.proj.weight.data.copy_(attn.out_proj.weight)
            new_attn.proj.bias.data.copy_(attn.out_proj.bias)
            set_module(block, 'self_attn', new_attn)
        o2 = fm.model.vision_model(debug_input).pooler_output
        
        # NOTE: bug is here!!!
        # although the diff is ZERO, but the logic of CLIPAttentionPrunable is incorrect!!!!
        diff = ((o1 - o2) ** 2).sum()
        print('diff before/after adding CLIPAttentionPrunable', diff)
        assert diff < 1e-4

        # print('\n\nDEBUG: WITHOUT ADDING CLIPAttentionPrunable\n\n')
        
        # exit()
        
        # return fm
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
            res = p_norm.argsort(descending=True)[0: int(n // reducing_width_ratio)].sort()[0]
            # print(res)
            return res
        
        # first_attn = True
        
        # for block_i, block in enumerate(fm_vit.model.text_model.encoder.layers):
        #     for k in ['k_proj', 'q_proj', 'v_proj']:
        #         qkv = get_module(block, f'self_attn.{k}')

        #         new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
        #                             qkv.bias is not None, qkv.weight.device)
        #         indexes = l1_max_indexes(qkv.weight.data, 0)
                
        #         new_qkv.weight.data.copy_(qkv.weight.data[indexes])
        #         if qkv.bias is not None:
        #             new_qkv.bias.data.copy_(qkv.bias.data[indexes])
        #         set_module(block, f'self_attn.{k}', new_qkv)

        #     proj = block.self_attn.out_proj
        #     new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
        #                         proj.bias is not None, proj.weight.device)
        #     new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
        #     if proj.bias is not None:
        #         new_proj.bias.data.copy_(proj.bias.data)
        #     set_module(block, f'self_attn.out_proj', new_proj)
            
        #     fc1 = block.mlp.fc1
        #     new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
        #                         fc1.bias is not None, fc1.weight.device)
        #     indexes = l1_max_indexes(fc1.weight.data, 0)
        #     new_fc1.weight.data.copy_(fc1.weight.data[indexes])
        #     if fc1.bias is not None:
        #         new_fc1.bias.data.copy_(fc1.bias.data[indexes])
        #     set_module(block, f'mlp.fc1', new_fc1)

        #     fc2 = block.mlp.fc2
        #     new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
        #                         fc2.bias is not None, fc2.weight.device)
        #     new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
        #     if fc2.bias is not None:
        #         new_fc2.bias.data.copy_(fc2.bias.data)
        #     set_module(block, f'mlp.fc2', new_fc2)
            
            
        for block_i, block in enumerate(fm_vit.model.vision_model.encoder.layers):
            # for k in ['k_proj', 'q_proj', 'v_proj']:
            #     qkv = get_module(block, f'self_attn.{k}')

            #     new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
            #                         qkv.bias is not None, qkv.weight.device)
            #     indexes = l1_max_indexes(qkv.weight.data, 0)
                
            #     new_qkv.weight.data.copy_(qkv.weight.data[indexes])
            #     if qkv.bias is not None:
            #         new_qkv.bias.data.copy_(qkv.bias.data[indexes])
            #     set_module(block, f'self_attn.{k}', new_qkv)

            # proj = block.self_attn.out_proj
            # new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
            #                     proj.bias is not None, proj.weight.device)
            # new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
            # if proj.bias is not None:
            #     new_proj.bias.data.copy_(proj.bias.data)
            # set_module(block, f'self_attn.out_proj', new_proj)
            
            
            # ------------------
            
            qkv = block.self_attn.qkv
            new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                qkv.bias is not None, qkv.weight.device)
            indexes = l1_max_indexes(qkv.weight.data, 0)
            
            new_qkv.weight.data.copy_(qkv.weight.data[indexes])
            if qkv.bias is not None:
                new_qkv.bias.data.copy_(qkv.bias.data[indexes])
            set_module(block, f'self_attn.qkv', new_qkv)
            proj = block.self_attn.proj
            new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
                                proj.bias is not None, proj.weight.device)
            new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
            if proj.bias is not None:
                new_proj.bias.data.copy_(proj.bias.data)
            set_module(block, f'self_attn.proj', new_proj)
            
            # --------------------
            
            fc1 = block.mlp.fc1
            new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
                                fc1.bias is not None, fc1.weight.device)
            indexes = l1_max_indexes(fc1.weight.data, 0)
            new_fc1.weight.data.copy_(fc1.weight.data[indexes])
            if fc1.bias is not None:
                new_fc1.bias.data.copy_(fc1.bias.data[indexes])
            set_module(block, f'mlp.fc1', new_fc1)

            fc2 = block.mlp.fc2
            new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
                                fc2.bias is not None, fc2.weight.device)
            new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
            if fc2.bias is not None:
                new_fc2.bias.data.copy_(fc2.bias.data)
            set_module(block, f'mlp.fc2', new_fc2)
            
        
        return fm_vit
    
    
    def init_md_from_fm_by_reducing_width_with_perf_test(self, fm: nn.Module, reducing_width_ratio: int,
                                                         samples: torch.Tensor) -> nn.Module:
        fm_size = get_model_size(fm, True)
        fm_latency = self._get_model_latency(fm, samples, 20, 
                                               get_model_device(fm), 20, False)
        
        master_dnn = self.init_md_from_fm_by_reducing_width(fm, reducing_width_ratio)
        master_dnn_size = get_model_size(master_dnn, True)
        logger.debug(f'inited master DNN: {master_dnn}')
        # from utils.dl.common.model import get_module
        # print('after generating')
        # get_module(fm, 'head').debug()
        # get_module(master_dnn, 'head').debug()
        # print('test master latency')
        master_dnn_latency = self._get_model_latency(master_dnn, samples, 20, 
                                               get_model_device(master_dnn), 20, False)

        logger.info(f'init master DNN (w/o FBS yet) by reducing foundation model\'s width (by {reducing_width_ratio:d}x)')
        logger.info(f'foundation model ({fm_size:.3f}MB, {fm_latency:.4f}s/sample) -> '
                    f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(fm_size / master_dnn_size):.2f}x, '
                    f'latency: ↓ {(fm_latency / master_dnn_latency):.2f}x)')
        
        return master_dnn
        
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