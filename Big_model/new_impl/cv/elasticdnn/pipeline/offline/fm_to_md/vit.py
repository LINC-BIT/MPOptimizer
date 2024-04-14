
import torch
from torch import nn 
from copy import deepcopy
import math
from .base import FM_to_MD_Util
from utils.common.log import logger
from utils.dl.common.model import set_module, get_module, get_super_module

from transformers.models.cvt.modeling_cvt import CvtSelfAttention,CvtAttention
from transformers import CvtConfig


class CvtSelfAttentionPrunable(CvtSelfAttention):
    def __init__(self,embed_dim,num_heads,with_cls_token):
        config = CvtConfig.from_pretrained('/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/cvt_model')
        
        super(CvtSelfAttentionPrunable, self).__init__(num_heads = num_heads,embed_dim = embed_dim,
                                                        kernel_size = 3,padding_q = config.padding_q,
                                                        padding_kv = config.padding_kv,stride_q = config.stride_q,
                                                        stride_kv = config.stride_kv,qkv_projection_method = config.qkv_projection_method,
                                                        qkv_bias = config.qkv_bias,attention_drop_rate = 0.0,with_cls_token=with_cls_token)

    # def transpose_for_scores(self, x):
    #     new_x_shape = x.size()[:-1] + (self.num_heads, -1)
    #     x = x.view(*new_x_shape)
    #     return x.permute(0, 2, 1, 3)
    
    def rearrange_for_multi_head_attention(self, hidden_state):
        batch_size, hidden_size, _ = hidden_state.shape
        head_dim = self.embed_dim // self.num_heads
        # rearrange 'b t (h d) -> b h t d'
        return hidden_state.view(batch_size, hidden_size, self.num_heads, -1).permute(0, 2, 1, 3)
    
    def forward(self, hidden_state, height, width):
        # mixed_query_layer = self.projection_query(hidden_states)
        # key_layer = self.transpose_for_scores(self.projection_key(hidden_states))
        # value_layer = self.transpose_for_scores(self.projection_value(hidden_states))
        # query_layer = self.transpose_for_scores(mixed_query_layer)

        # # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # #attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # if attention_mask is not None:
        #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        #     attention_scores = attention_scores + attention_mask

        # # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # # Mask heads if we want to
        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask

        # context_layer = torch.matmul(attention_probs, value_layer)

        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        # context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # return outputs
        if self.with_cls_token:
            cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
        batch_size, hidden_size, num_channels = hidden_state.shape
        # rearrange "b (h w) c -> b c h w"

        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        key = self.convolution_projection_key(hidden_state)
        query = self.convolution_projection_query(hidden_state)
        value = self.convolution_projection_value(hidden_state)

        if self.with_cls_token:
            query = torch.cat((cls_token, query), dim=1)
            key = torch.cat((cls_token, key), dim=1)
            value = torch.cat((cls_token, value), dim=1)

        head_dim = self.embed_dim // self.num_heads
        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        attention_score = torch.einsum("bhlk,bhtk->bhlt", [query, key]) * self.scale
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.einsum("bhlt,bhtv->bhlv", [attention_probs, value])
        # rearrange"b h t d -> b t (h d)"
        _, _, hidden_size, _ = context.shape
        #context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, hidden_size, self.num_heads * head_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, hidden_size, -1)
        return context
    @staticmethod
    def init_from_exist_self_attn(attn: CvtSelfAttention,embed_dim: float,num_heads,cls_token):
        # print(attn)
        
        res = CvtSelfAttentionPrunable(embed_dim,num_heads,with_cls_token = cls_token)
        for attr in dir(attn):
            # if str(attr) in ['transpose_for_scores'] or str(attr).startswith('_'):
            #     continue
            # if isinstance(getattr(attn, attr), nn.Module):
                # print(attr)
               
            if isinstance(getattr(attn, attr), nn.Module):
                try:
                    #print(attr, 'ok')
                    setattr(res, attr, getattr(attn, attr))
                    
                except Exception as e:
                    print(attr, str(e))
        
        return res


class FM_to_MD_ViT_Util(FM_to_MD_Util):
    def init_md_from_fm_by_reducing_width(self, fm: nn.Module, reducing_width_ratio: int) -> nn.Module:
        fm_vit = deepcopy(fm)
        config = [64,192,384]
        num_heads = [1,3,6]
        cls_token = [False,False,True]
        for block_1_i,block_1 in enumerate(fm_vit.cvt.encoder.stages):
            
            for block_i, block in enumerate(block_1.layers):
                set_module(block, 'attention.attention', CvtSelfAttentionPrunable.init_from_exist_self_attn(block.attention.attention,config[block_1_i],num_heads[block_1_i],cls_token=cls_token[block_1_i]))
        
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
        # print(fm_vit)
        for block_1_i,block_1 in enumerate(fm_vit.cvt.encoder.stages):
            
            for block_i, block in enumerate(block_1.layers):
                
                for k in ['projection_query','projection_key','projection_value']:
                    qkv = get_module(block, f'attention.attention.{k}')
                    #print(qkv.weight.data.shape)
                    new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                    qkv.bias is not None, qkv.weight.device)
                    indexes = l1_max_indexes(qkv.weight.data, 0)

                    new_qkv.weight.data.copy_(qkv.weight.data[indexes])
                    if qkv.bias is not None:
                        new_qkv.bias.data.copy_(qkv.bias.data[indexes])
                    set_module(block, f'attention.attention.{k}', new_qkv)
                
                # proj_1 = get_module(block, f'attention.output')  

                # new_proj_1 = nn.Linear(_f(proj.in_features), _f(proj.out_features), 
                #                 proj.bias is not None, proj.weight.device) 
                # new_proj_1.weight.data.copy_(proj_1.weight.data[:, l1_max_indexes(proj_1.weight.data, 1)])
                # if proj_1.bias is not None:
                #     new_proj_1.bias.data.copy_(proj_1.bias.data)   
                # set_module(block, f'attention.output', new_proj_1)   
                                         
                proj = get_module(block, f'attention.output.dense')
                    
                new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
                                proj.bias is not None, proj.weight.device)
                new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
                #new_proj.weight.data.copy_(proj.weight.data[l1_max_indexes(proj.weight.data, 1)])
                if proj.bias is not None:
                    new_proj.bias.data.copy_(proj.bias.data)
                set_module(block, f'attention.output.dense', new_proj)
            
                fc1 = get_module(block, f'intermediate.dense')
                
                new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
                                fc1.bias is not None, fc1.weight.device)
                indexes = l1_max_indexes(fc1.weight.data, 0)
                new_fc1.weight.data.copy_(fc1.weight.data[indexes])
                if fc1.bias is not None:
                    new_fc1.bias.data.copy_(fc1.bias.data[indexes])
                set_module(block, f'intermediate.dense', new_fc1)

                fc2 = get_module(block, f'output.dense')

                new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
                                fc2.bias is not None, fc2.weight.device)
                new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
                #new_fc2.weight.data.copy_(fc2.weight.data[l1_max_indexes(fc2.weight.data, 0)])
                if fc2.bias is not None:
                    new_fc2.bias.data.copy_(fc2.bias.data)
                set_module(block, f'output.dense', new_fc2)
        return fm_vit
        # for block_i, block in enumerate(fm_vit.cvt.encoder.stages.layers):
                
                
        #     for k in ['projection_query','projection_key','projection_value']:
        #         qkv = get_module(block, f'attention.attention.{k}')

        #         new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
        #                             qkv.bias is not None, qkv.weight.device)
        #         indexes = l1_max_indexes(qkv.weight.data, 0)
                
        #         new_qkv.weight.data.copy_(qkv.weight.data[indexes])
        #         if qkv.bias is not None:
        #                 new_qkv.bias.data.copy_(qkv.bias.data[indexes])
        #         set_module(block, f'attention.attention.{k}', new_qkv)
                
        #         # proj_1 = get_module(block, f'attention.output')  

        #         # new_proj_1 = nn.Linear(_f(proj.in_features), _f(proj.out_features), 
        #         #                 proj.bias is not None, proj.weight.device) 
        #         # new_proj_1.weight.data.copy_(proj_1.weight.data[:, l1_max_indexes(proj_1.weight.data, 1)])
        #         # if proj_1.bias is not None:
        #         #     new_proj_1.bias.data.copy_(proj_1.bias.data)   
        #         # set_module(block, f'attention.output', new_proj_1)   
                                         
        #     proj = get_module(block, f'attention.output.dense')
                    
        #     new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
        #                         proj.bias is not None, proj.weight.device)
        #     new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
        #     if proj.bias is not None:
        #             new_proj.bias.data.copy_(proj.bias.data)
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
        #                             fc2.bias is not None, fc2.weight.device)
        #     new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
        #     if fc2.bias is not None:
        #         new_fc2.bias.data.copy_(fc2.bias.data)
        #     set_module(block, f'output.dense', new_fc2)
        # return fm_vit
        # for block_i, block in enumerate(fm_vit.blocks):
        #     qkv = block.attn.qkv
            
        #     new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
        #                         qkv.bias is not None, qkv.weight.device)
        #     indexes = l1_max_indexes(qkv.weight.data, 0)
            
        #     new_qkv.weight.data.copy_(qkv.weight.data[indexes])
        #     if qkv.bias is not None:
        #         new_qkv.bias.data.copy_(qkv.bias.data[indexes])
        #     set_module(fm_vit, f'blocks.{block_i}.attn.qkv', new_qkv)

        #     proj = block.attn.proj
        #     new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
        #                         proj.bias is not None, proj.weight.device)
        #     new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
        #     if proj.bias is not None:
        #         new_proj.bias.data.copy_(proj.bias.data)
        #     set_module(fm_vit, f'blocks.{block_i}.attn.proj', new_proj)
            
        #     fc1 = block.mlp.fc1
        #     new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
        #                         fc1.bias is not None, fc1.weight.device)
        #     indexes = l1_max_indexes(fc1.weight.data, 0)
        #     new_fc1.weight.data.copy_(fc1.weight.data[indexes])
        #     if fc1.bias is not None:
        #         new_fc1.bias.data.copy_(fc1.bias.data[indexes])
        #     set_module(fm_vit, f'blocks.{block_i}.mlp.fc1', new_fc1)

        #     fc2 = block.mlp.fc2
        #     new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
        #                         fc2.bias is not None, fc2.weight.device)
        #     new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
        #     if fc2.bias is not None:
        #         new_fc2.bias.data.copy_(fc2.bias.data)
        #     set_module(fm_vit, f'blocks.{block_i}.mlp.fc2', new_fc2)
        #     # reduce dim_embedding
        #     # if name.endswith('patch_embed.proj'):
        #     #     continue
                
        #     #     new_layer = nn.Conv2d(module.in_channels, _f(module.out_channels), module.kernel_size, module.stride,
        #     #                          module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode,
        #     #                          module.weight.device)
                
        #     #     rand_indexes = l1_max_indexes(module.weight.data)
        #     #     new_layer.weight.data.copy_(module.weight.data[rand_indexes])
        #     #     if new_layer.bias is not None:
        #     #         new_layer.bias.data.copy_(module.bias.data[rand_indexes])
                
        #     #     fm_vit.cls_token.data = fm_vit.cls_token.data[:, :, rand_indexes]
        #     #     fm_vit.pos_embed.data = fm_vit.pos_embed.data[:, :, rand_indexes]
            
        #     # elif isinstance(module, nn.Linear):
                
        #     #     if 'head' in name:
        #     #         continue
                
        #     #         new_layer = nn.Linear(_f(module.in_features), module.out_features, 
        #     #                             module.bias is not None, module.weight.device)
        #     #         new_layer.weight.data.copy_(module.weight.data[:, l1_max_indexes(module.weight.data, 1)])
        #     #         if new_layer.bias is not None:
        #     #             new_layer.bias.data.copy_(module.bias.data)
        #     #     else:
        #     #         first_attn = False
        #     #         if first_attn:
        #     #             first_attn = False
        #     #             new_layer = nn.Linear(module.in_features, _f(module.out_features), 
        #     #                                 module.bias is not None, module.weight.device)
                        
        #     #             rand_indexes = l1_max_indexes(module.weight.data)
        #     #             new_layer.weight.data.copy_(module.weight.data[rand_indexes])
        #     #             if new_layer.bias is not None:
        #     #                 new_layer.bias.data.copy_(module.bias.data[rand_indexes])
        #     #         else:
        #     #             new_layer = nn.Linear(_f(module.in_features), _f(module.out_features), 
        #     #                                 module.bias is not None, module.weight.device)
                        
        #     #             rand_indexes = l1_max_indexes(module.weight.data)
        #     #             new_layer.weight.data.copy_(module.weight.data[rand_indexes][:, l1_max_indexes(module.weight.data, 1)])
        #     #             if new_layer.bias is not None:
        #     #                 new_layer.bias.data.copy_(module.bias.data[rand_indexes])
                        
        #     # elif isinstance(module, nn.LayerNorm) and ('block' in name or name == 'norm' or name == 'norm.0'):
        #     #     new_layer = nn.LayerNorm(_f(module.normalized_shape[0]), eps=module.eps, device=module.weight.device)
        #     #     rand_indexes = l1_max_indexes(module.weight.data)
        #     #     new_layer.weight.data.copy_(module.weight.data[rand_indexes])
        #     #     new_layer.bias.data.copy_(module.bias.data[rand_indexes])
                
        #     # else:
        #     #     continue
            
        #     # original_layer_str = str(module)
        #     # set_module(fm_vit, name, new_layer)
        #     # logger.debug(f'set_module, {name}, {new_layer}')
        #     # logger.debug(f'slim {name} from {original_layer_str} to {new_layer}')
        
        # return fm_vit