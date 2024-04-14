import torch
from torch import nn
from abc import ABC, abstractmethod

from utils.dl.common.model import get_model_device, get_model_latency, get_model_size
from utils.common.log import logger


class LoRA(nn.Linear):
    pass


class FMLoRA_Util(ABC):
    @abstractmethod
    def add_lora_ab_to_fm(self, fm: nn.Module, ab_r: int, samples: torch.Tensor):
        """
        only applying LoRA to attention weights.
        """
        pass
    
    def train_only_lora(self, fm: nn.Module):
        res = []
        for n, m in fm.named_modules():
            if isinstance(m, LoRA):
                for p in m.parameters():
                    p.requires_grad = True
                    res += [p]
            else:
                for p in m.parameters():
                    p.requires_grad = False
        return res
    
    @abstractmethod
    def absorb_lora_and_recover_net_structure(self, fm: nn.Module):
        pass
    