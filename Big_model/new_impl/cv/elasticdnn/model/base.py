import torch
from torch import nn
from abc import ABC, abstractmethod

from utils.dl.common.model import get_model_device, get_model_latency, get_model_size
from utils.common.log import logger


class KTakesAll(nn.Module):
    # k means sparsity (the larger k is, the smaller model is)
    def __init__(self, k):
        super(KTakesAll, self).__init__()
        self.k = k
        self.cached_i = None
        
    def forward(self, g: torch.Tensor):
        # k = int(g.size(1) * self.k)
        # i = (-g).topk(k, 1)[1]
        # t = g.scatter(1, i, 0)
        
        k = int(g.size(-1) * self.k)
        i = (-g).topk(k, -1)[1]
        self.cached_i = i
        t = g.scatter(-1, i, 0)
        
        return t
    
    
class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()
        
    def forward(self, x):
        return x.abs()


class Layer_WrappedWithFBS(nn.Module):
    def __init__(self):
        super(Layer_WrappedWithFBS, self).__init__()
        
        init_sparsity = 0.5
        self.k_takes_all = KTakesAll(init_sparsity)
        
        self.cached_raw_channel_attention = None
        self.cached_channel_attention = None
        self.use_cached_channel_attention = False


class ElasticDNNUtil(ABC):
    @abstractmethod
    def convert_raw_dnn_to_master_dnn(self, raw_dnn: nn.Module, r: float, ignore_layers=[]):
        raise NotImplementedError
    
    def convert_raw_dnn_to_master_dnn_with_perf_test(self, raw_dnn: nn.Module, r: float, ignore_layers=[]):
        raw_dnn_size = get_model_size(raw_dnn, True)
        master_dnn = self.convert_raw_dnn_to_master_dnn(raw_dnn, r, ignore_layers)
        master_dnn_size = get_model_size(master_dnn, True)
        
        logger.info(f'master DNN w/o FBS ({raw_dnn_size:.3f}MB) -> master DNN w/ FBS ({master_dnn_size:.3f}MB) '
                    f'(↑ {(((master_dnn_size - raw_dnn_size) / raw_dnn_size) * 100.):.2f}%)')
        return master_dnn
    
    def set_master_dnn_inference_via_cached_channel_attention(self, master_dnn: nn.Module):
        for name, module in master_dnn.named_modules():
            if isinstance(module, Layer_WrappedWithFBS):
                assert module.cached_channel_attention is not None
                module.use_cached_channel_attention = True
    
    def set_master_dnn_dynamic_inference(self, master_dnn: nn.Module):
        for name, module in master_dnn.named_modules():
            if isinstance(module, Layer_WrappedWithFBS):
                module.cached_channel_attention = None
                module.use_cached_channel_attention = False
    
    def train_only_fbs_of_master_dnn(self, master_dnn: nn.Module):
        fbs_params = []
        for n, p in master_dnn.named_parameters():
            if '.fbs' in n:
                fbs_params += [p]
                p.requires_grad = True
            else:
                p.requires_grad = False
        return fbs_params
    
    def get_accu_l1_reg_of_raw_channel_attention_in_master_dnn(self, master_dnn: nn.Module):
        res = 0.
        for name, module in master_dnn.named_modules():
            if isinstance(module, Layer_WrappedWithFBS):
                res += module.cached_raw_channel_attention.norm(1)
        return res
    
    def get_raw_channel_attention_in_master_dnn(self, master_dnn: nn.Module):
        res = {}
        for name, module in master_dnn.named_modules():
            if isinstance(module, Layer_WrappedWithFBS):
                res[name] = module.cached_raw_channel_attention
        return res

    def set_master_dnn_sparsity(self, master_dnn: nn.Module, sparsity: float):
        assert 0 <= sparsity <= 1., sparsity
        for name, module in master_dnn.named_modules():
            if isinstance(module, KTakesAll):
                module.k = sparsity
        logger.debug(f'set master DNN sparsity to {sparsity}')
        
    def clear_cached_channel_attention_in_master_dnn(self, master_dnn: nn.Module):
        for name, module in master_dnn.named_modules():
            if isinstance(module, Layer_WrappedWithFBS):
                module.cached_raw_channel_attention = None
                module.cached_channel_attention = None
                
    @abstractmethod
    def select_most_rep_sample(self, master_dnn: nn.Module, samples: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def extract_surrogate_dnn_via_samples(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):
        raise NotImplementedError
    
    def extract_surrogate_dnn_via_samples_with_perf_test(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):
        master_dnn_size = get_model_size(master_dnn, True)
        master_dnn_latency = get_model_latency(master_dnn, (1, *list(samples.size())[1:]), 50, 
                                               get_model_device(master_dnn), 50, False)
        
        res = self.extract_surrogate_dnn_via_samples(master_dnn, samples, return_detail)
        if not return_detail:
            surrogate_dnn = res
        else:
            surrogate_dnn, unpruned_indexes_of_layers = res
        surrogate_dnn_size = get_model_size(surrogate_dnn, True)
        surrogate_dnn_latency = get_model_latency(surrogate_dnn, (1, *list(samples.size())[1:]), 50, 
                                                  get_model_device(surrogate_dnn), 50, False)

        logger.info(f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample) -> '
                    f'surrogate DNN ({surrogate_dnn_size:.3f}MB, {surrogate_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(master_dnn_size / surrogate_dnn_size):.2f}x, '
                    f'latency: ↓ {(master_dnn_latency / surrogate_dnn_latency):.2f}x)')
        
        return res
