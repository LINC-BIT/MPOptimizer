from typing import Callable, Optional, Tuple, Union
from timm.layers import Mlp, PatchEmbed
from timm.models.vision_transformer import Block, VisionTransformer
from .head import DecoderLinear


class ViTForSeg(VisionTransformer):
    def __init__(self, img_size: int | Tuple[int, int] = 224, patch_size: int | Tuple[int, int] = 16, in_chans: int = 3, num_classes: int = 1000, global_pool: str = 'token', embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4, qkv_bias: bool = True, qk_norm: bool = False, init_values: float | None = None, class_token: bool = True, no_embed_class: bool = False, pre_norm: bool = False, fc_norm: bool | None = None, drop_rate: float = 0, pos_drop_rate: float = 0, patch_drop_rate: float = 0, proj_drop_rate: float = 0, attn_drop_rate: float = 0, drop_path_rate: float = 0, weight_init: str = '', embed_layer: Callable[..., Any] = ..., norm_layer: Callable[..., Any] | None = None, act_layer: Callable[..., Any] | None = None, block_fn: Callable[..., Any] = ..., mlp_layer: Callable[..., Any] = ...):
        super().__init__(img_size, patch_size, in_chans, num_classes, global_pool, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_norm, init_values, class_token, no_embed_class, pre_norm, fc_norm, drop_rate, pos_drop_rate, patch_drop_rate, proj_drop_rate, attn_drop_rate, drop_path_rate, weight_init, embed_layer, norm_layer, act_layer, block_fn, mlp_layer)
        self.head = DecoderLinear(num)
    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x)

    def init_from_vit(self, vit):
        self.load_state_dict(vit.state_dict(), strict=False)
