from typing import Optional
import torch
from copy import deepcopy
from torch import nn
from utils.common.others import get_cur_time_str
from utils.dl.common.model import get_model_device, get_model_latency, get_model_size, get_module, get_super_module, set_module
from utils.common.log import logger
from utils.third_party.nni_new.compression.pytorch.speedup import ModelSpeedup
import os

from .base import Abs, KTakesAll, Layer_WrappedWithFBS, ElasticDNNUtil


class Conv2d_WrappedWithFBS(Layer_WrappedWithFBS):
    def __init__(self, raw_conv2d: nn.Conv2d, raw_bn: nn.BatchNorm2d, r):
        super(Conv2d_WrappedWithFBS, self).__init__()
        
        self.fbs = nn.Sequential(
            Abs(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(raw_conv2d.in_channels, raw_conv2d.out_channels // r),
            nn.ReLU(),
            nn.Linear(raw_conv2d.out_channels // r, raw_conv2d.out_channels),
            nn.ReLU()
        )
        
        self.raw_conv2d = raw_conv2d
        self.raw_bn = raw_bn # remember clear the original BNs in the network
        
        nn.init.constant_(self.fbs[5].bias, 1.)
        nn.init.kaiming_normal_(self.fbs[5].weight)

    def forward(self, x):
        raw_x = self.raw_bn(self.raw_conv2d(x))
        
        if self.use_cached_channel_attention and self.cached_channel_attention is not None:
            channel_attention = self.cached_channel_attention
        else:
            self.cached_raw_channel_attention = self.fbs(x)
            self.cached_channel_attention = self.k_takes_all(self.cached_raw_channel_attention)
            
            channel_attention = self.cached_channel_attention
        
        return raw_x * channel_attention.unsqueeze(2).unsqueeze(3)
    
    
class StaticFBS(nn.Module):
    def __init__(self, channel_attention: torch.Tensor):
        super(StaticFBS, self).__init__()
        assert channel_attention.dim() == 1
        self.channel_attention = nn.Parameter(channel_attention.unsqueeze(0).unsqueeze(2).unsqueeze(3), requires_grad=False)
        
    def forward(self, x):
        return x * self.channel_attention
    
    def __str__(self) -> str:
        return f'StaticFBS({len(self.channel_attention.size(1))})'
    
    
class ElasticCNNUtil(ElasticDNNUtil):
    def convert_raw_dnn_to_master_dnn(self, raw_dnn: nn.Module, r: float, ignore_layers=[]):
        model = deepcopy(raw_dnn)

        # clear original BNs
        num_original_bns = 0
        last_conv_name = None
        conv_bn_map = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name = name
            if isinstance(module, nn.BatchNorm2d) and (ignore_layers is not None and last_conv_name not in ignore_layers):
                num_original_bns += 1
                conv_bn_map[last_conv_name] = name
        
        num_conv = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and (ignore_layers is not None and name not in ignore_layers):
                set_module(model, name, Conv2d_WrappedWithFBS(module, get_module(model, conv_bn_map[name]), r))
                num_conv += 1
                
        assert num_conv == num_original_bns
        
        for bn_layer in conv_bn_map.values():
            set_module(model, bn_layer, nn.Identity())
            
        return model
    
    def select_most_rep_sample(self, master_dnn: nn.Module, samples: torch.Tensor):
        return samples[0].unsqueeze(0)
    
    def extract_surrogate_dnn_via_samples(self, master_dnn: nn.Module, samples: torch.Tensor):
        sample = self.select_most_rep_sample(master_dnn, samples)
        assert sample.dim() == 4 and sample.size(0) == 1
        
        master_dnn.eval()
        with torch.no_grad():
            master_dnn_output = master_dnn(sample)
        
        pruning_info = {}
        pruning_masks = {}
        
        for layer_name, layer in master_dnn.named_modules():
            if not isinstance(layer, Conv2d_WrappedWithFBS):
                continue
            
            cur_pruning_mask = {'weight': torch.zeros_like(layer.raw_conv2d.weight.data)}
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'] = torch.zeros_like(layer.raw_conv2d.bias.data)
            
            w = get_module(master_dnn, layer_name).cached_channel_attention.squeeze(0)
            unpruned_filters_index = w.nonzero(as_tuple=True)[0]
            pruning_info[layer_name] = w
            
            cur_pruning_mask['weight'][unpruned_filters_index, ...] = 1.
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'][unpruned_filters_index, ...] = 1.
            pruning_masks[layer_name + '.0'] = cur_pruning_mask
        
        surrogate_dnn = deepcopy(master_dnn)
        for name, layer in surrogate_dnn.named_modules():
            if not isinstance(layer, Conv2d_WrappedWithFBS):
                continue
            set_module(surrogate_dnn, name, nn.Sequential(layer.raw_conv2d, layer.raw_bn, nn.Identity()))
            
        # fixed_pruning_masks = fix_mask_conflict(pruning_masks, fbs_model, sample.size(), None, True, True, True)
        tmp_mask_path = f'tmp_mask_{get_cur_time_str()}_{os.getpid()}.pth'
        torch.save(pruning_masks, tmp_mask_path)
        surrogate_dnn.eval()
        model_speedup = ModelSpeedup(surrogate_dnn, sample, tmp_mask_path, sample.device)
        model_speedup.speedup_model()
        os.remove(tmp_mask_path)
        
        # add feature boosting module
        for layer_name, feature_boosting_w in pruning_info.items():
            feature_boosting_w = feature_boosting_w[feature_boosting_w.nonzero(as_tuple=True)[0]]
            set_module(surrogate_dnn, layer_name + '.2', StaticFBS(feature_boosting_w))
            
        surrogate_dnn.eval()
        with torch.no_grad():
            surrogate_dnn_output = surrogate_dnn(sample)
        output_diff = ((surrogate_dnn_output - master_dnn_output) ** 2).sum()
        assert output_diff < 1e-4, output_diff
        logger.info(f'output diff of master and surrogate DNN: {output_diff}')
        
        return surrogate_dnn
        