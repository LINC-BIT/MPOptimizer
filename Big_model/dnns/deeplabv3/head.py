from torch import nn 
from einops import rearrange
import torch.nn.functional as F

from utils.dl.common.model import get_super_module


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder, im_size):
        super(DecoderLinear, self).__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.im_size = im_size

        self.head = nn.Linear(self.d_encoder, n_cls)
        
    def debug(self):
        print(self.head, id(self), 'debug()')

    def forward(self, x):
        # print('inside debug')
        # self.debug()
        x = x[:, 1:] # remove cls token
        # print(x.size())
        
        H, W = self.im_size
        GS = H // self.patch_size
        # print(H, W, GS, self.patch_size)
        # print('head', self.head.weight.size(), x.size())
        # print(self.head, 'debug()')
        x = self.head(x)
        # print(x.size())
        
        # (b, HW//ps**2, ps_c)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)
        
        # print(x.size())
        
        masks = x
        masks = F.upsample(masks, size=(H, W), mode="bilinear")
        
        # print(masks.size())

        return masks
    
    
def modify_forward_head():
    from types import MethodType
    from timm.models.vision_transformer import VisionTransformer
    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x)
    VisionTransformer.forward_head = MethodType(forward_head, VisionTransformer)
