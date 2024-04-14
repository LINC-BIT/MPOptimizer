from typing import List
from data.dataloader import build_dataloader
# from methods.elasticdnn.api.online_model import ElasticDNN_OnlineModel
from methods.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel

import torch
import sys
from torch import nn
from methods.elasticdnn.api.model import ElasticDNN_OfflineSegFMModel, ElasticDNN_OfflineSegMDModel
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.vit import FMLoRA_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.common.file import ensure_dir
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F
from utils.dl.common.env import create_tbwriter
import os
from utils.common.log import logger
from utils.common.data_record import write_json
# from methods.shot.shot import OnlineShotModel
from methods.ewc.ewc_elasticfm import OnlineEWCModel
import tqdm
# from methods.feat_align.mmd import mmd_rbf
from copy import deepcopy


class ElasticDNN_ClsOnlineModel(ElasticDNN_OnlineModel):
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
            
            if 'head' in p_name:
                continue
            # if False:
                # self.last_trained_cls_indexes 
                assert hasattr(self, 'last_trained_cls_indexes')
                print(self.last_trained_cls_indexes)

                diff = self._compute_diff(matched_md_param, p)
                # matched_md_param[self.last_trained_cls_indexes].copy_(p[self.last_trained_cls_indexes.to(self.device)])
                matched_md_param.copy_(p)
                logger.debug(f'SPECIFIC FOR CL HEAD | end feedback: {p_name}, diff: {diff:.6f}')
            else:
                diff = self._compute_diff(matched_md_param, (matched_md_param + p) / 2.)
                matched_md_param.copy_((matched_md_param + p) / 2.)
                logger.debug(f'end feedback: {p_name}, diff: {diff:.6f}')
            
    def add_cls_in_head(self, num_cls):
        head: nn.Linear = get_module(self.models_dict['md'], 'head')
        
        new_head = nn.Linear(head.in_features, head.out_features + num_cls, head.bias is not None, device=self.device)
        
        # nn.init.zeros_(new_head.weight.data)
        # nn.init.zeros_(new_head.bias.data)
        
        new_head.weight.data[0: head.out_features] = deepcopy(head.weight.data)
        new_head.bias.data[0: head.out_features] = deepcopy(head.bias.data)
        set_module(self.models_dict['md'], 'head', new_head)
        set_module(self.models_dict['fm'], 'head', new_head)
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)
                output = self.infer(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticViTUtil()
    
    def get_fm_matched_param_of_md_param(self, md_param_name):
        # only between qkv.weight, norm.weight/bias
        self_param_name = md_param_name
        fm = self.models_dict['fm']
        if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
            return None
        
        p = get_parameter(self.models_dict['md'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1 and 'norm' in self_param_name and 'weight' in self_param_name:
            return get_parameter(fm, self_param_name)
        
        # if 'head' in self_param_name:
        #     return get_parameter(fm, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if 'qkv.weight' in self_param_name:
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.qkv'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.abs'
            fm_abs = get_module(fm, fm_abs_name)
            
            # NOTE: unrecoverable operation! multiply LoRA parameters to allow it being updated in update_fm_param()
            # TODO: if fm will be used for inference, _mul_lora_weight will not be applied!
            if not hasattr(fm_abs, '_mul_lora_weight'):
                logger.debug(f'set _mul_lora_weight in {fm_abs_name}')
                setattr(fm_abs, '_mul_lora_weight', 
                        nn.Parameter(torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0)))
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs._mul_lora_weight.data # task-specific params (LoRA)
            ], dim=0)
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
            
        elif 'mlp.fc1' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)

        elif 'mlp.fc2' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name
            return get_parameter(fm, fm_param_name)
        
        else:
            # return get_parameter(fm, self_param_name)
            return None
        
    def update_fm_param(self, md_param_name, cal_new_fm_param_by_md_param):
        if not 'qkv.weight' in md_param_name:
            matched_fm_param_ref = self.get_fm_matched_param_of_md_param(md_param_name)
            matched_fm_param_ref.copy_(cal_new_fm_param_by_md_param)
        else:
            new_fm_attn_weight, new_fm_lora_weight = torch.chunk(cal_new_fm_param_by_md_param, 2, 0)
            
            ss = md_param_name.split('.')
            fm = self.models_dict['fm']
            
            # update task-agnostic parameters
            fm_qkv_name = '.'.join(ss[0: -1]) + '.qkv'
            fm_qkv = get_module(fm, fm_qkv_name)
            fm_qkv.weight.data.copy_(new_fm_attn_weight)
            
            # update task-specific parameters
            fm_abs_name = '.'.join(ss[0: -1]) + '.abs'
            fm_abs = get_module(fm, fm_abs_name)
            fm_abs._mul_lora_weight.data.copy_(new_fm_lora_weight) # TODO: this will not be applied in inference!
        
    def get_md_matched_param_of_fm_param(self, fm_param_name):
        return super().get_md_matched_param_of_fm_param(fm_param_name)
    
    def get_md_matched_param_of_sd_param(self, sd_param_name):
        # only between qkv.weight, norm.weight/bias
        self_param_name = sd_param_name
        md = self.models_dict['md']
        if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
            return None
        
        p = get_parameter(self.models_dict['sd'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1 and 'norm' in self_param_name and 'weight' in self_param_name:
            return get_parameter(md, self_param_name)
        
        if 'head' in self_param_name:
            return get_parameter(md, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if 'qkv.weight' in self_param_name:
            return get_parameter(md, self_param_name)
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
            
        elif 'mlp.fc1.0.weight' in self_param_name:
            fm_param_name = '.'.join(self_param_name.split('.')[0: -2]) + '.linear.weight'
            return get_parameter(md, fm_param_name)

        elif 'mlp.fc2' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name
            return get_parameter(md, fm_param_name)
        
        else:
            # return get_parameter(fm, self_param_name)
            return None
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['sd'], 'head')
        return list(head.parameters())
    
    
    
    
from typing import List, Tuple
from data.dataloader import build_dataloader
# from methods.elasticdnn.api.online_model import ElasticDNN_OnlineModel
from methods.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel

import torch
import sys
from torch import nn
from methods.elasticdnn.api.model import ElasticDNN_OfflineSegFMModel, ElasticDNN_OfflineSegMDModel
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.vit import FMLoRA_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.common.file import ensure_dir
from utils.dl.common.model import LayerActivation, LayerActivation2, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F
from utils.dl.common.env import create_tbwriter
import os
from utils.common.log import logger
from utils.common.data_record import write_json
# from methods.shot.shot import OnlineShotModel
from methods.gem.gem_el import OnlineGEMModel
import tqdm
from methods.feat_align.mmd import mmd_rbf
from copy import deepcopy


class ClsOnlineGEMModel(OnlineGEMModel):
    def get_trained_params(self):
        qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters() if 'qkv.weight' in n or 'norm' in n or 'mlp' in n]
        return qkv_and_norm_params
    
    def forward_to_get_task_loss(self, x, y):
        return F.cross_entropy(self.infer(x), y)
    
    def add_cls_in_head(self, num_cls):
        return

        head: nn.Linear = get_module(self.models_dict['main'], 'head')
        
        new_head = nn.Linear(head.in_features, head.out_features + num_cls, head.bias is not None, device=self.device)
        new_head.weight.data[0: head.out_features] = deepcopy(head.weight.data)
        new_head.bias.data[0: head.out_features] = deepcopy(head.bias.data)
        set_module(self.models_dict['main'], 'head', new_head)
        
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](x)
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        _d = test_loader.dataset
        from data import build_dataloader, split_dataset
        if _d.__class__.__name__ == '_SplitDataset' and _d.underlying_dataset.__class__.__name__ == 'MergedDataset': # necessary for CL
            print('\neval on merged datasets')
            
            merged_full_dataset = _d.underlying_dataset.datasets
            ratio = len(_d.keys) / len(_d.underlying_dataset)
            
            if int(len(_d) * ratio) == 0:
                ratio = 1.
            # print(ratio)
            # bs = 
            # test_loaders = [build_dataloader(split_dataset(d, min(max(test_loader.batch_size, int(len(d) * ratio)), len(d)))[0], # TODO: this might be overlapped with train dataset
            #                                  min(test_loader.batch_size, int(len(d) * ratio)), 
            #                                  test_loader.num_workers, False, None) for d in merged_full_dataset]

            test_loaders = []
            for d in merged_full_dataset:
                n = int(len(d) * ratio)
                if n == 0:
                    n = len(d)
                sub_dataset = split_dataset(d, min(max(test_loader.batch_size, n), len(d)))[0]
                loader = build_dataloader(sub_dataset, min(test_loader.batch_size, n), test_loader.num_workers, False, None)
                test_loaders += [loader]
            
            accs = [self.get_accuracy(loader) for loader in test_loaders]
            print(accs)
            return sum(accs) / len(accs)
        
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)
                output = self.infer(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc