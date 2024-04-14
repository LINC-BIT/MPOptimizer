from transformers import BlipForQuestionAnswering, BlipConfig,BlipModel,CLIPVisionModel,CLIPConfig
import torch
from torch import nn
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import tqdm
import torch.nn.functional as F
from utils.dl.common.model import get_model_device, get_model_latency, get_model_size, set_module
from utils.dl.common.model import set_module, get_module, get_super_module
from utils.common.log import logger
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util, LoRA
from transformers.models.blip.modeling_blip import BlipAttention
#from transformers.models.blip.modeling_blip_text import BlipTextSelfAttention,BlipTextAttention,BlipTextSelfOutput
from transformers.models.beit.modeling_beit import BeitSelfAttention,BeitConfig
from transformers.models.clip.modeling_clip import CLIPAttention
from new_impl.cv.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from new_impl.cv.elasticdnn.model.base import Abs, KTakesAll, ElasticDNNUtil, Layer_WrappedWithFBS

from typing import Optional, Tuple
import math

from transformers.models.sam.modeling_sam import SamVisionEncoder,SamPreTrainedModel,SamAttention,SamConfig,SamVisionAttention
from new_impl.cv.dnns.deeplabv3.head import DecoderLinear
config = SamConfig.from_pretrained('new_impl/cv/sam/sam_pretrained')

class Sammodel(SamPreTrainedModel):
    def __init__(self, config,num_classes):
        config = SamConfig.from_pretrained('new_impl/cv/sam/sam_pretrained')
        super(Sammodel,self).__init__(config)
        self.vision_encoder = SamVisionEncoder(config.vision_config)
        self.head = DecoderLinear(num_classes, 16, 256, (224, 224)).to('cuda')
    def forward(self,x):
        x = self.vision_encoder(x)
        x = rearrange(x[0],'b c h w  -> b (h w) c')
        output = self.head(x)
        return output
# def blip(num_classes):
#     model =  BlipForQuestionAnswering.from_pretrained('new_impl/mm/Vis_bert/QuestionAnswering/VisBert_pretrained')
#     # linear  = model.text_decoder.cls.predictions.decoder
#     # new_linear = nn.Linear(linear.in_features,30524,bias = True)
#     # set_module(model,'text_decoder.cls.predictions.decoder',new_linear)
#     return model
# def blip(num_classes):
#     model = BlipForQuestionAnswering.from_pretrained('new_impl/mm/Vis_bert/QuestionAnswering/VisBert_pretrained')
#     linear = model.text_decoder.cls.predictions.decoder
#     new_linear = nn.Linear(linear.in_features,num_classes,bias = True)
#     set_module(model,'text_decoder.cls.predictions.decoder',new_linear)
#     return model
# class clip(nn.Module):
#     def __init__(self,num_classes):
#         super(clip,self).__init__()
#         self.clip = CLIPVisionModel.from_pretrained('new_impl/cv/clip/pretrained_model')
#         self.classifier = nn.Linear(768,num_classes)

#     def forward(self,sample):
#         output = self.clip(sample)[-1]#output the last hidden
#         output  = self.classifier(output)
#         return output

    
class ToQKV_WrappedWithLoRA(nn.Module):
    def __init__(self, fc: nn.Linear, ab_r: int):
        super(ToQKV_WrappedWithLoRA, self).__init__()
        
        self.fc = fc
        self.ab = self.create_ab_as_linear(fc.weight.data, ab_r)
        
    def create_ab_as_linear(self, fc_weight: torch.Tensor, ab_r: int):
        res = nn.Sequential(
            LoRA(fc_weight.size(1), fc_weight.size(0) // ab_r, bias=False),
            LoRA(fc_weight.size(0) // ab_r, fc_weight.size(0), bias=False)
        ).to(fc_weight.device)
        nn.init.kaiming_uniform_(res[0].weight, a=5 ** 0.5)
        nn.init.zeros_(res[1].weight)
        return res
        
    def forward(self, x):
        x1 = self.fc(x)
        x2 = self.ab(x)
        return x1 + x2
    

class FMLoRA_sam_Util(FMLoRA_Util):
    
    @torch.no_grad()
    def add_lora_ab_to_fm(self, fm: nn.Module, ab_r: int, samples: torch.Tensor):
        fm.eval()
        
        # print(samples)
        # for k, v in samples.items():
        #     if isinstance(v, torch.Tensor):
        #         samples[k] = v.to(get_model_device(fm))
        
        #o1 = fm.generate(**samples)
        o1 = fm(samples)
        for name, module in fm.named_modules():
            if name.endswith(('query', 'key', 'value')):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
            elif name.endswith('.qkv'):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
            elif name.endswith(('k_proj','q_proj','v_proj')):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))


        #o2 = fm.generate(**samples)
        o2 = fm(samples)
        if isinstance(o1, tuple):
            o1 = o1[-1]
            o2 = o2[-1]
        output_diff = ((o1 - o2) ** 2).sum()
        assert output_diff < 1e-5
        return fm
    
    @torch.no_grad()
    def absorb_lora_and_recover_net_structure(self, fm: nn.Module, samples: torch.Tensor):       
        fm.eval()
        # print('absorb lora before')

        # for k, v in samples.items():
        #     if isinstance(v, torch.Tensor):
        #         samples[k] = v.to(get_model_device(fm))
        
        o1 = fm(samples)
        
        for name, module in fm.named_modules():
            if not isinstance(module, ToQKV_WrappedWithLoRA):
                continue
            
            fc = module.fc
            ab = module.ab

            fc.weight.add_(ab[1].weight @ ab[0].weight)
            
            set_module(fm, name, fc)
        
        # print('absorb lora after')
        o2 = fm(samples)
        
        if isinstance(o1, tuple):
            o1 = o1[-1]
            o2 = o2[-1]
        output_diff = ((o1 - o2) ** 2).sum()
        assert output_diff < 1e-6, output_diff
        
        return fm
    

####Here start with Fbs

# class blipTextAttentionPrunable(BlipTextSelfAttention):
#     def __init__(self,is_cross_attention):
#         config = BlipConfig.from_pretrained('new_impl/mm/Vis_bert/QuestionAnswering/VisBert_pretrained')
#         super(blipTextAttentionPrunable,self).__init__(config.text_config,is_cross_attention)    
        
#     def save_attn_gradients(self, attn_gradients):
#         self.attn_gradients = attn_gradients

#     def get_attn_gradients(self):
#         return self.attn_gradients

#     def save_attention_map(self, attention_map):
#         self.attention_map = attention_map

#     def get_attention_map(self):
#         return self.attention_map

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.FloatTensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.Tensor]:
#         mixed_query_layer = self.query(hidden_states)

#         # If this is instantiated as a cross-attention module, the keys
#         # and values come from an encoder; the attention mask needs to be
#         # such that the encoder's padding tokens are not attended to.
#         is_cross_attention = encoder_hidden_states is not None

#         if is_cross_attention:
#             key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
#             value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
#             attention_mask = encoder_attention_mask
#         elif past_key_value is not None:
#             key_layer = self.transpose_for_scores(self.key(hidden_states))
#             value_layer = self.transpose_for_scores(self.value(hidden_states))
#             key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
#             value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
#         else:
#             key_layer = self.transpose_for_scores(self.key(hidden_states))
#             value_layer = self.transpose_for_scores(self.value(hidden_states))

#         query_layer = self.transpose_for_scores(mixed_query_layer)

#         past_key_value = (key_layer, value_layer)

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

#         if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
#             seq_length = hidden_states.size()[1]
#             position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
#             position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
#             distance = position_ids_l - position_ids_r
#             positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
#             positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

#             if self.position_embedding_type == "relative_key":
#                 relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
#                 attention_scores = attention_scores + relative_position_scores
#             elif self.position_embedding_type == "relative_key_query":
#                 relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
#                 relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
#                 attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         if attention_mask is not None:
#             # Apply the attention mask is (precomputed for all layers in BlipTextModel forward() function)
#             attention_scores = attention_scores + attention_mask.to(attention_scores.device)

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs_dropped = self.dropout(attention_probs)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attention_probs_dropped = attention_probs_dropped * head_mask

#         context_layer = torch.matmul(attention_probs_dropped, value_layer)

#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (-1,)
#         context_layer = context_layer.view(*new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

#         outputs = outputs + (past_key_value,)
#         return outputs
#     @staticmethod
#     def init_from_exist_self_attn(attn: BlipTextSelfAttention,is_cross_attention):
#         # print(attn)
        
#         res = blipTextAttentionPrunable(is_cross_attention)
        
#         for attr in dir(attn):
#             # if str(attr) in ['transpose_for_scores'] or str(attr).startswith('_'):
#             #     continue
#             # if isinstance(getattr(attn, attr), nn.Module):
#                 # print(attr)
                
#             if isinstance(getattr(attn, attr), nn.Module):
#                 try:
#                     # print(attr, 'ok')
#                     setattr(res, attr, getattr(attn, attr))
                    
#                 except Exception as e:
#                     print(attr, str(e))
        
        
        
#         return res
    

    
# class blipSelfTextAttentionPrunable(BlipTextAttention):
#     def __init__(self, config, is_cross_attention=False):
#         self.self = blipTextAttentionPrunable(config, is_cross_attention)
#         self.output = BlipTextSelfOutput(config)
#         self.pruned_heads = set()
#         super(blipSelfTextAttentionPrunable,self).__init__(config)

#     def prune_heads(self, heads):
#         if len(heads) == 0:
#             return
#         heads, index = find_pruneable_heads_and_indices(
#             heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
#         )

#         # Prune linear layers
#         self.self.query = prune_linear_layer(self.self.query, index)
#         self.self.key = prune_linear_layer(self.self.key, index)
#         self.self.value = prune_linear_layer(self.self.value, index)
#         self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

#         # Update hyper params and store pruned heads
#         self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
#         self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
#         self.pruned_heads = self.pruned_heads.union(heads)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.FloatTensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.Tensor]:
#         self_outputs = self.self(
#             hidden_states,
#             attention_mask,
#             head_mask,
#             encoder_hidden_states,
#             encoder_attention_mask,
#             past_key_value,
#             output_attentions,
#         )
#         attention_output = self.output(self_outputs[0], hidden_states)
#         outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
#         return outputs
#     @staticmethod
#     def init_from_exist_self_attn(attn: BlipTextAttention,config,is_cross_attention):
#         # print(attn)
        
#         res = blipTextAttentionPrunable(config,is_cross_attention)
        
#         for attr in dir(attn):
#             # if str(attr) in ['transpose_for_scores'] or str(attr).startswith('_'):
#             #     continue
#             # if isinstance(getattr(attn, attr), nn.Module):
#                 # print(attr)
                
#             if isinstance(getattr(attn, attr), nn.Module):
#                 try:
#                     # print(attr, 'ok')
#                     setattr(res, attr, getattr(attn, attr))
                    
#                 except Exception as e:
#                     print(attr, str(e))
        
        
        
#         return res
    





# class blipSelfAttentionPrunable(BlipAttention):
#     def __init__(self):
#         config = BlipConfig.from_pretrained('new_impl/mm/Vis_bert/QuestionAnswering/VisBert_pretrained')
#         super(blipSelfAttentionPrunable, self).__init__(config.vision_config)

#     def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
#         return tensor.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2).contiguous()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         """Input shape: Batch x Time x Channel"""

#         bsz, tgt_len, embed_dim = hidden_states.size()

#         mixed_qkv = (
#             self.qkv(hidden_states)
#             .reshape(bsz, tgt_len, 3, self.num_heads, -1)
#             .permute(2, 0, 3, 1, 4)
#         )
#         query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

#         attention_scores = attention_scores * self.scale

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.functional.softmax(attention_scores, dim=-1)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask

#         context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

#         new_context_layer_shape = context_layer.size()[:-2] + (-1,)
#         context_layer = context_layer.reshape(new_context_layer_shape)

#         output = self.projection(context_layer)

#         outputs = (output, attention_probs) if output_attentions else (output, None)

#         return outputs
    
#     @staticmethod
#     def init_from_exist_self_attn(attn: BlipAttention):
#         # print(attn)
        
#         res = blipSelfAttentionPrunable()
        
#         for attr in dir(attn):
#             # if str(attr) in ['transpose_for_scores'] or str(attr).startswith('_'):
#             #     continue
#             # if isinstance(getattr(attn, attr), nn.Module):
#                 # print(attr)
                
#             if isinstance(getattr(attn, attr), nn.Module):
#                 try:
#                     # print(attr, 'ok')
#                     setattr(res, attr, getattr(attn, attr))
                    
#                 except Exception as e:
#                     print(attr, str(e))
        
        
        
#         return res
# class CLIPAttentionPrunable(CLIPAttention):
#     def __init__(self, config: CLIPConfig, ratio:int) -> None:
#         config = CLIPConfig.from_pretrained('new_impl/cv/clip/pretrained_model')
#         super(CLIPAttentionPrunable, self).__init__(config.vision_config)
#         self.head_dim = self.head_dim // ratio
        
#     def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
#         return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         causal_attention_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         """Input shape: Batch x Time x Channel"""

#         bsz, tgt_len, embed_dim = hidden_states.size()

#         # get query proj
#         query_states = self.q_proj(hidden_states) * self.scale
#         key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
#         value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

#         proj_shape = (bsz * self.num_heads, -1, self.head_dim)
#         query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
#         key_states = key_states.view(*proj_shape)
#         value_states = value_states.view(*proj_shape)

#         src_len = key_states.size(1)
#         attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

#         if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
#                 f" {attn_weights.size()}"
#             )

#         # apply the causal_attention_mask first
#         if causal_attention_mask is not None:
#             if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
#                     f" {causal_attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, tgt_len, src_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)

#         if output_attentions:
#             # this operation is a bit akward, but it's required to
#             # make sure that attn_weights keeps its gradient.
#             # In order to do so, attn_weights have to reshaped
#             # twice and have to be reused in the following
#             attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
#         else:
#             attn_weights_reshaped = None

#         attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

#         attn_output = torch.bmm(attn_probs, value_states)

#         if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
#         attn_output = attn_output.transpose(1, 2)
#         attn_output = attn_output.reshape(bsz, tgt_len, -1)

#         attn_output = self.out_proj(attn_output)

#         return attn_output, attn_weights_reshaped
    
#     @staticmethod
#     def init_from_exist_self_attn(attn: CLIPAttention,config,ratio:int):
#         # print(attn)
        
#         res = CLIPAttentionPrunable(config,ratio)
        
#         for attr in dir(attn):
#             # if str(attr) in ['transpose_for_scores'] or str(attr).startswith('_'):
#             #     continue
#             # if isinstance(getattr(attn, attr), nn.Module):
#                 # print(attr)
                
#             if isinstance(getattr(attn, attr), nn.Module):
#                 try:
#                     # print(attr, 'ok')
#                     setattr(res, attr, getattr(attn, attr))
                    
#                 except Exception as e:
#                     print(attr, str(e))
#         return res

class SamVisionAttentionPrunable(SamVisionAttention):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config,ratio):
        super(SamVisionAttentionPrunable,self).__init__(config,window_size=14)
        window_size = 14
        head_dim = config.hidden_size // config.num_attention_heads
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim//ratio))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim//ratio))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """

        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )

        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        # print(reshaped_query.shape)
        # print(relative_position_height.shape)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)

        return outputs
    @staticmethod
    def init_from_exist_self_attn(attn: SamVisionAttention,config,ratio):
        # print(attn)
        
        res = SamVisionAttentionPrunable(config,ratio)
        
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

class FM_to_MD_sam_Util(FM_to_MD_Util):
    def init_md_from_fm_by_reducing_width(self, fm: nn.Module, reducing_width_ratio: int) -> nn.Module:
        fm_vis = deepcopy(fm)
        config = SamConfig.from_pretrained('new_impl/cv/sam/sam_pretrained')

        for block_i,block in enumerate(fm_vis.vision_encoder.layers):
            set_module(block, 'attn', SamVisionAttentionPrunable.init_from_exist_self_attn(block.attn,config.vision_config,reducing_width_ratio))

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
        
        for block_i, block in enumerate(fm_vis.vision_encoder.layers):
            
            qkv = get_module(block, f'attn.qkv')

            new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                    qkv.bias is not None, qkv.weight.device)
            indexes = l1_max_indexes(qkv.weight.data, 0)
                
            new_qkv.weight.data.copy_(qkv.weight.data[indexes])
            if qkv.bias is not None:
                new_qkv.bias.data.copy_(qkv.bias.data[indexes])
            set_module(block, f'attn.qkv', new_qkv)
            
            proj = get_module(block, f'attn.proj')
            new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
                                proj.bias is not None, proj.weight.device)
            new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
            if proj.bias is not None:
                new_proj.bias.data.copy_(proj.bias.data)
            set_module(block, f'attn.proj', new_proj)
            
            fc1 = get_module(block, f'mlp.lin1')
            new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
                                fc1.bias is not None, fc1.weight.device)
            indexes = l1_max_indexes(fc1.weight.data, 0)
            new_fc1.weight.data.copy_(fc1.weight.data[indexes])
            if fc1.bias is not None:
                new_fc1.bias.data.copy_(fc1.bias.data[indexes])
            set_module(block, f'mlp.lin1', new_fc1)

            fc2 = get_module(block, f'mlp.lin2')
            new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
                                fc2.bias is not None, fc2.weight.device)
            new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
            if fc2.bias is not None:
                new_fc2.bias.data.copy_(fc2.bias.data)
            set_module(block, f'mlp.lin2', new_fc2)


        # for block_i, block in enumerate(fm_vis.text_decoder.bert.encoder.layer):
        #     for k in ['query', 'key', 'value']:
        #         qkv = get_module(block, f'crossattention.self.{k}')

        #         new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
        #                             qkv.bias is not None, qkv.weight.device)
        #         indexes = l1_max_indexes(qkv.weight.data, 0)
                
        #         new_qkv.weight.data.copy_(qkv.weight.data[indexes])
        #         if qkv.bias is not None:
        #             new_qkv.bias.data.copy_(qkv.bias.data[indexes])
        #         set_module(block, f'crossattention.self.{k}', new_qkv)
            
        #     proj = get_module(block, f'crossattention.output.dense')
        #     new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
        #                         proj.bias is not None, proj.weight.device)
        #     new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
        #     if proj.bias is not None:
        #         new_proj.bias.data.copy_(proj.bias.data)
        #     set_module(block, f'crossattention.output.dense', new_proj)
            
        
        # for block_i, block in enumerate(fm_vis.text_encoder.encoder.layer):
        #     for k in ['query', 'key', 'value']:
        #         qkv = get_module(block, f'attention.self.{k}')

        #         new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
        #                             qkv.bias is not None, qkv.weight.device)
        #         indexes = l1_max_indexes(qkv.weight.data, 0)
                
        #         new_qkv.weight.data.copy_(qkv.weight.data[indexes])
        #         if qkv.bias is not None:
        #             new_qkv.bias.data.copy_(qkv.bias.data[indexes])
        #         set_module(block, f'attention.self.{k}', new_qkv)
            
        #     proj = get_module(block, f'attention.output.dense')
        #     new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
        #                         proj.bias is not None, proj.weight.device)
        #     new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
        #     if proj.bias is not None:
        #         new_proj.bias.data.copy_(proj.bias.data)
        #     set_module(block, f'attention.output.dense', new_proj)
            
        #     fc1 = get_module(block, f'intermediate.dense')
        #     new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
        #                         fc1.bias is not None, fc1.weight.device)
        #     indexes = l1_max_indexes(fc1.weight.data, 0)
        #     new_fc1.weight.data.copy_(fc1.weight.data[indexes])
        #     if fc1.bias is not None:
        #         new_fc1.bias.data.copy_(fc1.bias.data[indexes])
        #     set_module(block, f'intermediate.dense', new_fc1)

        #     fc2 = get_module(block, f'output.dense')
        #     new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
        #                         fc2.bias is not None, fc2.weight.device)
        #     new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
        #     if fc2.bias is not None:
        #         new_fc2.bias.data.copy_(fc2.bias.data)
        #     set_module(block, f'output.dense', new_fc2)


        # for block_i, block in enumerate(fm_vis.text_encoder.encoder.layer):
        #     for k in ['query', 'key', 'value']:
        #         qkv = get_module(block, f'crossattention.self.{k}')

        #         new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
        #                             qkv.bias is not None, qkv.weight.device)
        #         indexes = l1_max_indexes(qkv.weight.data, 0)
                
        #         new_qkv.weight.data.copy_(qkv.weight.data[indexes])
        #         if qkv.bias is not None:
        #             new_qkv.bias.data.copy_(qkv.bias.data[indexes])
        #         set_module(block, f'crossattention.self.{k}', new_qkv)
            
        #     proj = get_module(block, f'crossattention.output.dense')
        #     new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
        #                         proj.bias is not None, proj.weight.device)
        #     new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
        #     if proj.bias is not None:
        #         new_proj.bias.data.copy_(proj.bias.data)
        #     set_module(block, f'crossattention.output.dense', new_proj)
            
        
        
        # for block_i, block in enumerate(fm_vis.vision_model.encoder.layers):
        #     qkv = block.self_attn.qkv
            
        #     new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
        #                         qkv.bias is not None, qkv.weight.device)
        #     indexes = l1_max_indexes(qkv.weight.data, 0)
            
        #     new_qkv.weight.data.copy_(qkv.weight.data[indexes])
        #     if qkv.bias is not None:
        #         new_qkv.bias.data.copy_(qkv.bias.data[indexes])
        #     set_module(fm_vis, f'vision_model.encoder.layers.{block_i}.self_attn.qkv', new_qkv)

        #     proj = block.self_attn.projection
        #     new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
        #                         proj.bias is not None, proj.weight.device)
        #     new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
        #     if proj.bias is not None:
        #         new_proj.bias.data.copy_(proj.bias.data)
        #     set_module(fm_vis, f'vision_model.encoder.layers.{block_i}.self_attn.projection', new_proj)
            
        #     fc1 = block.mlp.fc1
        #     new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
        #                         fc1.bias is not None, fc1.weight.device)
        #     indexes = l1_max_indexes(fc1.weight.data, 0)
        #     new_fc1.weight.data.copy_(fc1.weight.data[indexes])
        #     if fc1.bias is not None:
        #         new_fc1.bias.data.copy_(fc1.bias.data[indexes])
        #     set_module(fm_vis, f'vision_model.encoder.layers.{block_i}.mlp.fc1', new_fc1)

        #     fc2 = block.mlp.fc2
        #     new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
        #                         fc2.bias is not None, fc2.weight.device)
        #     new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
        #     if fc2.bias is not None:
        #         new_fc2.bias.data.copy_(fc2.bias.data)
        #     set_module(fm_vis, f'vision_model.encoder.layers.{block_i}.mlp.fc2', new_fc2)

        return fm_vis
    
    def init_md_from_fm_by_reducing_width_with_perf_test(self, fm: nn.Module, reducing_width_ratio: int,
                                                         samples: torch.Tensor) -> nn.Module:
        fm_size = get_model_size(fm, True)
        fm_latency = self._get_model_latency(fm, samples, 20, 
                                               get_model_device(fm), 20, False)
        
        master_dnn = self.init_md_from_fm_by_reducing_width(fm, reducing_width_ratio)
        master_dnn_size = get_model_size(master_dnn, True)
        logger.debug(f'inited master DNN: {master_dnn}')
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
                model(dummy_input)
                
        infer_time_list = []
                
        if device == 'cuda' or 'cuda' in str(device):
            with torch.no_grad():
                for _ in range(sample_num):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    model(dummy_input)
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


####Here starts with index

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
            #Rearrange('b n d -> b d n'),
            Rearrange('b h w d-> b d (h w)'),
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
        raw_res = rearrange(raw_res,'b h w d-> b (h w) d')
        res = channel_attention.unsqueeze(1) * raw_res
        res = rearrange(res,'b (h w) d->b h w d',h=14,w=14)
        return res
    
    
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
    
    
class ElasticsamUtil(ElasticDNNUtil):
    def convert_raw_dnn_to_master_dnn(self, raw_dnn: nn.Module, r: float, ignore_layers=[]):
        assert len(ignore_layers) == 0, 'not supported yet'

        raw_vit = deepcopy(raw_dnn)
        
        # set_module(module, 'patch_embed.proj', ProjConv_WrappedWithFBS(module.patch_embed.proj, r))
                
        for name, module in raw_vit.named_modules():
            # if name.endswith('attn'):
            #     set_module(module, 'qkv', ToQKV_WrappedWithFBS(module.qkv, r))
            if name.endswith('intermediate'):
                set_module(module, 'dense', Linear_WrappedWithFBS(module.dense, r))
            elif name.endswith('mlp'):
                set_module(module, 'lin1', Linear_WrappedWithFBS(module.lin1, r))
        
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
        return samples[0].unsqueeze(0)
        # res = {k: v[0: 1] for k, v in samples.items()}
        # return res
        
    def extract_surrogate_dnn_via_samples(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):#产生小模型的步骤
        sample = self.select_most_rep_sample(master_dnn, samples)
        # assert sample.dim() == 4 and sample.size(0) == 1
        
        # print('before')
        master_dnn.eval()
        self.clear_cached_channel_attention_in_master_dnn(master_dnn)
        with torch.no_grad():
            master_dnn_output = master_dnn(sample)
            
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
        for block_i, block in enumerate(boosted_vit.vision_encoder.layers):
            # attn = block.attn
            # ff = block.mlp
            
            ff_0 = get_module(block, f'mlp.lin1')
            # ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, k)
            ff_0_pruned_indexes = ff_0.k_takes_all.cached_i[0].sort()[0]
            ff_0_unpruned_indexes = torch.LongTensor([ii for ii in range(ff_0.cached_channel_attention.size(1)) if ii not in ff_0_pruned_indexes])
            new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
            new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
            if ff_0.linear.bias is not None:
                new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
            set_module(block, 'mlp.lin1', nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
            
            ff_1 = get_module(block, f'mlp.lin2')
            new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
            new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
            if ff_1.bias is not None:
                new_ff_1.bias.data.copy_(ff_1.bias.data)
            set_module(block, 'mlp.lin2', new_ff_1)
            
            unpruned_indexes_of_layers[f'vision_encoder.layers.{block_i}.mlp.lin1.0.weight'] = ff_0_unpruned_indexes
        # for block_i,block in enumerate(boosted_vit.vision_model.encoder.layers):

        #     attn = block.self_attn
        #     ff = block.mlp
        #     ff_0 = ff.fc1
        #     # ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, k)
        #     ff_0_pruned_indexes = ff_0.k_takes_all.cached_i[0].sort()[0]
        #     ff_0_unpruned_indexes = torch.LongTensor([ii for ii in range(ff_0.cached_channel_attention.size(1)) if ii not in ff_0_pruned_indexes])
        #     new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
        #     new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
        #     if ff_0.linear.bias is not None:
        #         new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
        #     set_module(ff, 'fc1', nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
            
        #     ff_1 = ff.fc2
        #     new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
        #     new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
        #     if ff_1.bias is not None:
        #         new_ff_1.bias.data.copy_(ff_1.bias.data)
        #     set_module(ff, 'fc2', new_ff_1)
            
        #     unpruned_indexes_of_layers[f'vision_model.encoder.layers.{block_i}.mlp.fc1.0.weight'] = ff_0_unpruned_indexes


        # for block_i, block in enumerate(boosted_vit.text_decoder.bert.encoder.layer):
        #     # attn = block.attn
        #     # ff = block.mlp
            
        #     ff_0 = get_module(block, f'intermediate.dense')
        #     # ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, k)
        #     ff_0_pruned_indexes = ff_0.k_takes_all.cached_i[0].sort()[0]
        #     ff_0_unpruned_indexes = torch.LongTensor([ii for ii in range(ff_0.cached_channel_attention.size(1)) if ii not in ff_0_pruned_indexes])
        #     new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
        #     new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
        #     if ff_0.linear.bias is not None:
        #         new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
        #     set_module(block, 'intermediate.dense', nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
            
        #     ff_1 = get_module(block, f'output.dense')
        #     new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
        #     new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
        #     if ff_1.bias is not None:
        #         new_ff_1.bias.data.copy_(ff_1.bias.data)
        #     set_module(block, 'output.dense', new_ff_1)
            
        #     unpruned_indexes_of_layers[f'text_decoder.bert.encoder.layer.{block_i}.intermediate.dense.0.weight'] = ff_0_unpruned_indexes
        surrogate_dnn = boosted_vit
        surrogate_dnn.eval()
        surrogate_dnn = surrogate_dnn.to(get_model_device(master_dnn))
        # logger.debug(surrogate_dnn)
        with torch.no_grad():
            surrogate_dnn_output = surrogate_dnn(sample)
            
        output_diff = ((surrogate_dnn_output - master_dnn_output) ** 2).sum()
        # assert output_diff < 1e-4, output_diff
        logger.info(f'output diff of master and surrogate DNN: {output_diff}')
        # logger.debug(f'example output of master/surrogate: {master_dnn_output.sum(0)[0: 10]}, {surrogate_dnn_output.sum(0)[0: 10]}')
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
                model(dummy_input)
                
        infer_time_list = []
                
        if device == 'cuda' or 'cuda' in str(device):
            with torch.no_grad():
                for _ in range(sample_num):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    model(dummy_input)
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



#####Here starts with online
