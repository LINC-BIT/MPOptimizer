from transformers import BlipForQuestionAnswering, BlipConfig,BlipModel, GitModel
import torch
from torch import nn
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import tqdm

from utils.dl.common.model import get_model_device, get_model_latency, get_model_size, set_module
from utils.dl.common.model import set_module, get_module, get_super_module
from utils.common.log import logger
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util, LoRA
from transformers.models.blip.modeling_blip import BlipAttention
from transformers.models.blip.modeling_blip_text import BlipTextSelfAttention,BlipTextAttention,BlipTextSelfOutput
from new_impl.cv.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from new_impl.cv.elasticdnn.model.base import Abs, KTakesAll, ElasticDNNUtil, Layer_WrappedWithFBS

from typing import Optional, Tuple
import math

class git(nn.Module):
    def __init__(self,num_classes):
        super(git,self).__init__()
        self.git =GitModel.from_pretrained('')
        self.cls = nn.Linear(768,num_classes)

    def forward(self,**sample):
        output = self.blip(**sample)[-1]#output the last hidden
        output  = self.cls(output[1])
        return output

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

class FMLoRA_git_Util(FMLoRA_Util):
    
    @torch.no_grad()
    def add_lora_ab_to_fm(self, fm: nn.Module, ab_r: int, samples: dict):
        fm.eval()
        
        # print(samples)
        for k, v in samples.items():
            if isinstance(v, torch.Tensor):
                samples[k] = v.to(get_model_device(fm))
        
        o1 = fm(**samples)
        #o1 = fm(**samples)
        for name, module in fm.named_modules():
            if name.endswith(('query', 'key', 'value')):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
            elif name.endswith('.qkv'):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))


        o2 = fm(**samples)
        #o2 = fm(**samples)
        if isinstance(o1, tuple):
            o1 = o1[-1]
            o2 = o2[-1]
        output_diff = ((o1 - o2) ** 2).sum()
        assert output_diff < 1e-5
        return fm
    
    @torch.no_grad()
    def absorb_lora_and_recover_net_structure(self, fm: nn.Module, samples: dict):       
        fm.eval()
        # print('absorb lora before')

        for k, v in samples.items():
            if isinstance(v, torch.Tensor):
                samples[k] = v.to(get_model_device(fm))
        
        o1 = fm(**samples)
        
        for name, module in fm.named_modules():
            if not isinstance(module, ToQKV_WrappedWithLoRA):
                continue
            
            fc = module.fc
            ab = module.ab

            fc.weight.add_(ab[1].weight @ ab[0].weight)
            
            set_module(fm, name, fc)
        
        # print('absorb lora after')
        o2 = fm(**samples)
        
        if isinstance(o1, tuple):
            o1 = o1[-1]
            o2 = o2[-1]
        output_diff = ((o1 - o2) ** 2).sum()
        assert output_diff < 1e-6, output_diff
        
        return fm
