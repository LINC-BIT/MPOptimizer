import torch
from torch import nn
from abc import ABC, abstractmethod

from utils.dl.common.model import get_model_device, get_model_latency, get_model_size
from utils.common.log import logger


class FM_to_MD_Util(ABC):
    """
    Foundation Model (FM) to Master DNN (MD), where MD is a narrower FM (with smaller width but the same depth).

    MD is pre-trained by knowledge distillation;
    Moreover, we construct the index relationship between FM and MD in this process, 
    enabling the lightweight knowledge feedback from MD to FM.
    
    NOTE: 索引建立在master DNN权重通道和LoRA的AB之间
    """

    @abstractmethod
    def init_md_from_fm_by_reducing_width(self, fm: nn.Module, reducing_width_ratio: int) -> nn.Module:
        raise NotImplementedError
    
    def init_md_from_fm_by_reducing_width_with_perf_test(self, fm: nn.Module, reducing_width_ratio: int,
                                                         samples: torch.Tensor) -> nn.Module:
        fm_size = get_model_size(fm, True)
        fm_latency = get_model_latency(fm, (1, *list(samples.size())[1:]), 20, 
                                               get_model_device(fm), 20, False)
        
        master_dnn = self.init_md_from_fm_by_reducing_width(fm, reducing_width_ratio)
        master_dnn_size = get_model_size(master_dnn, True)
        logger.debug(f'inited master DNN: {master_dnn}')
        #print(master_dnn)
        # from utils.dl.common.model import get_module
        # print('after generating')
        # get_module(fm, 'head').debug()
        # get_module(master_dnn, 'head').debug()
        # print('test master latency')
        master_dnn_latency = get_model_latency(master_dnn, (1, *list(samples.size())[1:]), 20, 
                                               get_model_device(master_dnn), 20, False)

        logger.info(f'init master DNN (w/o FBS yet) by reducing foundation model\'s width (by {reducing_width_ratio:d}x)')
        logger.info(f'foundation model ({fm_size:.3f}MB, {fm_latency:.4f}s/sample) -> '
                    f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(fm_size / master_dnn_size):.2f}x, '
                    f'latency: ↓ {(fm_latency / master_dnn_latency):.2f}x)')
        
        return master_dnn
        