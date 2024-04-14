from copy import deepcopy
from typing import List
import torch
from new_impl.cv.base.model import BaseModel
import tqdm
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod
from new_impl.cv.elasticdnn.model.base import ElasticDNNUtil
from new_impl.cv.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from utils.common.log import logger
from utils.dl.common.model import LayerActivation, get_parameter


class ElasticDNN_OnlineModel(BaseModel):
    def __init__(self, name: str, models_dict_path: str, device: str, ab_options: dict):
        super().__init__(name, models_dict_path, device)
        
        assert [k in ab_options.keys() for k in ['md_to_fm_alpha', 'fm_to_md_alpha']]
        self.ab_options = ab_options
        
    def get_required_model_components(self) -> List[str]:
        return ['fm', 'md', 'sd', 'indexes', 'bn_stats']
    
    @torch.no_grad()
    def generate_sd_by_target_samples(self, target_samples: torch.Tensor):
        elastic_dnn_util = self.get_elastic_dnn_util()
        
        if isinstance(target_samples, dict):
            for k, v in target_samples.items():
                if isinstance(v, torch.Tensor):
                    target_samples[k] = v.to(self.device)
        else:
            target_samples = target_samples.to(self.device)
        sd, unpruned_indexes_of_layers = elastic_dnn_util.extract_surrogate_dnn_via_samples_with_perf_test(self.models_dict['md'], target_samples, True)
        logger.debug(f'generate sd: \n{sd}')
        return sd, unpruned_indexes_of_layers
    
    @torch.no_grad()
    def _compute_diff(self, old, new):
        return (new - old).norm(1) / old.norm(1)
    
    @torch.no_grad()
    def sd_feedback_to_md(self, after_da_sd, unpruned_indexes_of_layers):
        self.models_dict['sd'] = after_da_sd
        self.before_da_md = deepcopy(self.models_dict['md'])
        
        logger.info('\n\nsurrogate DNN feedback to master DNN...\n\n')
        # one-to-one
        
        cur_unpruned_indexes = None
        cur_unpruned_indexes_name = None
        
        for p_name, p in self.models_dict['sd'].named_parameters():
            matched_md_param = self.get_md_matched_param_of_sd_param(p_name)
            logger.debug(f'if feedback: {p_name}')
            if matched_md_param is None:
                continue
            logger.debug(f'start feedback: {p_name}, {p.size()} -> {matched_md_param.size()}')
            # average
            # setattr(matched_md_module, matched_md_param_name, (matched_md_param + p) / 2.)
            
            if p_name in unpruned_indexes_of_layers.keys():
                cur_unpruned_indexes = unpruned_indexes_of_layers[p_name]
                cur_unpruned_indexes_name = p_name
            
            if p.size() != matched_md_param.size():
                logger.debug(f'cur unpruned indexes: {cur_unpruned_indexes_name}, {cur_unpruned_indexes.size()}')
                
                if p.dim() == 1: # norm
                    new_p = deepcopy(matched_md_param)
                    new_p[cur_unpruned_indexes] = p
                elif p.dim() == 2: # linear
                    if p.size(0) < matched_md_param.size(0): # output pruned
                        new_p = deepcopy(matched_md_param)
                        new_p[cur_unpruned_indexes] = p
                    else: # input pruned
                        new_p = deepcopy(matched_md_param)
                        new_p[:, cur_unpruned_indexes] = p
                p = new_p
                
            assert p.size() == matched_md_param.size(), f'{p.size()}, {matched_md_param.size()}'

            diff = self._compute_diff(matched_md_param, (matched_md_param + p) / 2.)
            matched_md_param.copy_((matched_md_param + p) / 2.)
            logger.debug(f'end feedback: {p_name}, diff: {diff:.6f}')

    def infer(self, x, *args, **kwargs):
        return self.models_dict['sd'](x)
    
    def set_sd_sparsity(self, sparsity: float):
        elastic_dnn_util = self.get_elastic_dnn_util()
        elastic_dnn_util.clear_cached_channel_attention_in_master_dnn(self.models_dict['md'])
        elastic_dnn_util.set_master_dnn_sparsity(self.models_dict['md'], sparsity)
    
    @torch.no_grad()
    def md_feedback_to_self_fm(self):
        logger.info('\n\nmaster DNN feedback to self foundation model...\n\n')
        # one-to-many
        
        # def upsample_2d_tensor(p: torch.Tensor, target_len: int):
        #     assert p.dim() == 2 # regard 2d weight as (batch_size, 1d_vector_dim)
        #     return F.upsample(p.unsqueeze(1).unsqueeze(3),
        #                     size=(target_len, 1),
        #                     mode='bilinear').squeeze(3).squeeze(1)
        
        for (p_name, p), before_p in zip(self.models_dict['md'].named_parameters(), self.before_da_md.parameters()):
            matched_fm_param = self.get_fm_matched_param_of_md_param(p_name)
            logger.debug(f'if feedback: {p_name}')
            if matched_fm_param is None:
                continue
            # print(self.models_dict['indexes'].keys())
            index = self.models_dict['indexes'][p_name]
            logger.debug(f'start feedback: {p_name}, {p.size()} -> {matched_fm_param.size()}, index: {index.size()}')
            
            p_update = p - before_p
            
            t = False
            if p.dim() > 1 and index.size(0) == p.size(1) and index.size(1) == matched_fm_param.size(1):
                assert p.dim() == 2
                p_update = p_update.T
                matched_fm_param = matched_fm_param.T
                t = True
                logger.debug(f'transpose paramters')
            
            
            if p.dim() == 2:
                # p_update = upsample_2d_tensor(p_update, matched_fm_param.size(1))
                
                p_update = p_update.unsqueeze(1)
                index = index.unsqueeze(-1)
                
                # fast
                # agg_p_update = (p_update * index).sum(0)
                
                # balanced agg
                agg_p_update = 0
            
                cur_split_size = 64
                while index.size(0) % cur_split_size != 0:
                    cur_split_size -= 1
                
                for i in range(0, index.size(0), cur_split_size):
                    agg_p_update += p_update[i: i + cur_split_size] * index[i: i + cur_split_size]
                agg_p_update = agg_p_update.sum(0)
                
                
            else:
                agg_p_update = (p_update.unsqueeze(1) * index).sum(0)
            
            new_fm_param = matched_fm_param + agg_p_update * self.ab_options['md_to_fm_alpha']
            
            diff = self._compute_diff(matched_fm_param, new_fm_param)
            
            # NOTE: matched_fm_param may not be reference, may be a deepcopy!!
            # and only here matched_fm_param needs to be updated, so another method dedicated for updating is necessary here
            # matched_fm_param.copy_(new_fm_param)
            self.update_fm_param(p_name, new_fm_param.T if t else new_fm_param)
            
            logger.debug(f'end feedback: {p_name}, diff: {diff:.6f} (md_to_fm_alpha={self.ab_options["md_to_fm_alpha"]:.4f})')
            
    @abstractmethod
    @torch.no_grad()
    def update_fm_param(self, md_param_name, cal_new_fm_param_by_md_param):
        """
        you should get the reference of fm_param and update it
        """
        raise NotImplementedError

    @torch.no_grad()
    def aggregate_fms_to_self_fm(self, fms: List[nn.Module]):
        # average task-agnositc parameters
        logger.info('\n\naggregate foundation models to self foundation model...\n\n')
        for p_name, self_p in self.models_dict['fm'].named_parameters():
            logger.debug(f'if aggregate {p_name}')
            if 'abs' in p_name or p_name.startswith('norm') or p_name.startswith('head'):
                logger.debug(f'{p_name} belongs to LoRA parameters/task-specific head, i.e. task-specific parameters, skip')
                continue
            all_p = [get_parameter(fm, p_name) for fm in fms]
            if any([_p is None for _p in all_p]):
                continue
            
            avg_p = sum(all_p) / len(all_p)
            # [_p.copy_(avg_p) for _p in all_p]
            
            diff = self._compute_diff(self_p, avg_p)
            logger.debug(f'aggregate {p_name}, diff {diff:.6f}')
            
            self_p.copy_(avg_p)
            
    @torch.no_grad()
    def fm_feedback_to_md(self):
        logger.info('\n\nself foundation model feedback to master DNN...\n\n')
        # one-to-many
        
        # def downsample_2d_tensor(p: torch.Tensor, target_len: int):
        #     assert p.dim() == 2 # regard 2d weight as (batch_size, 1d_vector_dim)
        #     # return F.upsample(p.unsqueeze(1).unsqueeze(3),
        #     #                 size=(target_len, 1),
        #     #                 mode='bilinear').squeeze(3).squeeze(1)
        #     return F.interpolate(p.unsqueeze(1).unsqueeze(3), size=(target_len, 1), mode='bilinear').squeeze(3).squeeze(1)
            
            
        for p_name, p in self.models_dict['md'].named_parameters():
            matched_fm_param = self.get_fm_matched_param_of_md_param(p_name)
            logger.debug(f'if feedback: {p_name}')
            if matched_fm_param is None:
                continue
            
            index = self.models_dict['indexes'][p_name]
            logger.debug(f'start feedback: {p_name}, {p.size()} -> {matched_fm_param.size()}, index: {index.size()}')
            
            if p.dim() > 1 and index.size(0) == p.size(1) and index.size(1) == matched_fm_param.size(1):
                assert p.dim() == 2
                p = p.T
                matched_fm_param = matched_fm_param.T
            
            if p.dim() == 2:
                # matched_fm_param = downsample_2d_tensor(matched_fm_param, p.size(1))
                
                matched_fm_param = matched_fm_param.unsqueeze(0)
                index = index.unsqueeze(-1)
                
                # fast
                # agg_p_update = (p_update * index).sum(0)
                
                # balanced agg
                agg_fm_param = 0
            
                cur_split_size = 64
                while index.size(1) % cur_split_size != 0:
                    cur_split_size -= 1
                
                for i in range(0, index.size(1), cur_split_size):
                    agg_fm_param += matched_fm_param[:, i: i + cur_split_size] * index[:, i: i + cur_split_size]
                agg_fm_param = agg_fm_param.sum(1)
                # agg_fm_param = downsample_2d_tensor(agg_fm_param, p.size(1))
                
            else:
                agg_fm_param = (matched_fm_param.unsqueeze(0) * index).sum(1)
                
            
            
            diff = self._compute_diff(p, agg_fm_param)
            p.copy_(agg_fm_param * self.ab_options['fm_to_md_alpha'] + (1. - self.ab_options['fm_to_md_alpha']) * p)
            
            logger.debug(f'end feedback: {p_name}, diff: {diff:.6f} (fm_to_md_alpha: {self.ab_options["fm_to_md_alpha"]:.4f})')
        
    @abstractmethod
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        pass
    
    @abstractmethod
    def get_task_head_params(self):
        pass
    
    @abstractmethod
    def get_md_matched_param_of_sd_param(self, sd_param_name):
        pass
    
    @abstractmethod
    def get_fm_matched_param_of_md_param(self, md_param_name):
        pass
    
    @abstractmethod
    def get_md_matched_param_of_fm_param(self, fm_param_name):
        pass