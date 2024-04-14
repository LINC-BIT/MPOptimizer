from typing import List
import torch
from methods.base.model import BaseModel
import tqdm
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util

from utils.dl.common.model import LayerActivation


class HuggingFaceModelAPI:
    @abstractmethod
    def get_feature_hook(self, fm: nn.Module) -> LayerActivation:
        pass
    
    @abstractmethod
    def get_task_head_params(self, fm: nn.Module):
        pass
    
    @abstractmethod
    def get_qkv_proj_ff1_ff2_layer_names(self):
        pass
    
    @abstractmethod
    def get_accuracy(self, fm: nn.Module, test_loader, device, *args, **kwargs):
        pass
    
    @abstractmethod
    def infer(self, fm: nn.Module, x, *args, **kwargs):
        pass
    
    @abstractmethod
    def forward_to_get_task_loss(self, fm: nn.Module, x, y, *args, **kwargs):
        pass