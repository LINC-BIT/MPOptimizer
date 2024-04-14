import torch
from torch import nn
from abc import ABC, abstractmethod

from utils.dl.common.model import get_model_device, get_model_latency, get_model_size, set_module
from utils.common.log import logger
from .base import FMLoRA_Util, LoRA


class ToQKV_WrappedWithLoRA(nn.Module):
    def __init__(self, qkv: nn.Linear, ab_r: int):
        super(ToQKV_WrappedWithLoRA, self).__init__()
        
        self.qkv = qkv
        self.abs = nn.ModuleList([self.create_ab_as_linear(w, ab_r) for w in qkv.weight.data.chunk(3, dim=0)])
        
    def create_ab_as_linear(self, fc_weight: torch.Tensor, ab_r: int):
        res = nn.Sequential(
            LoRA(fc_weight.size(1), fc_weight.size(0) // ab_r, bias=False),
            LoRA(fc_weight.size(0) // ab_r, fc_weight.size(0), bias=False)
        ).to(fc_weight.device)
        nn.init.kaiming_uniform_(res[0].weight, a=5 ** 0.5)
        nn.init.zeros_(res[1].weight)
        return res
        
    def forward(self, x):
        x1 = self.qkv(x)
        x2 = torch.cat([ab(x) for ab in self.abs], dim=-1)
        return x1 + x2
    

class FMLoRA_ViT_Util(FMLoRA_Util):
    
    @torch.no_grad()
    def add_lora_ab_to_fm(self, fm: nn.Module, ab_r: int, samples: torch.Tensor):
        fm.eval()
        o1 = fm(samples)
        
        for name, module in fm.named_modules():
            if not name.endswith('.qkv'):
                continue    
            set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
            
        o2 = fm(samples)
        
        output_diff = ((o1 - o2) ** 2).sum()
        assert output_diff < 1e-5
        
        return fm
    
    @torch.no_grad()
    def absorb_lora_and_recover_net_structure(self, fm: nn.Module, samples: torch.Tensor):
        fm.eval()
        # print('absorb lora before')
        o1 = fm(samples)
        
        for name, module in fm.named_modules():
            if not isinstance(module, ToQKV_WrappedWithLoRA):
                continue
            
            qkv = module.qkv
            fm_abs = module.abs

            fm_abs_weight = torch.cat([_abs[1].weight @ _abs[0].weight for _abs in fm_abs], dim=0)
            qkv.weight.add_(fm_abs_weight)
            
            set_module(fm, name, qkv)
        
        # print('absorb lora after')
        o2 = fm(samples)
        
        output_diff = ((o1 - o2) ** 2).sum()
        # print(o1)
        # print(o2)
        assert output_diff < 1e-5, output_diff
        
        return fm
    
    
