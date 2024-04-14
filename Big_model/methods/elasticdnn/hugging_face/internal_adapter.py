from ..api.model import ElasticDNN_OfflineFMModel, ElasticDNN_OfflineMDModel
from .user_impl import HuggingFaceModelAPI

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
from methods.elasticdnn.model.bert import ElasticBertUtil
from utils.common.file import ensure_dir
from utils.dl.common.model import LayerActivation, get_module, get_parameter, get_super_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F
from utils.dl.common.env import create_tbwriter
import os
from utils.common.log import logger
from utils.common.data_record import write_json
# from methods.shot.shot import OnlineShotModel
from methods.feat_align.main import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from methods.feat_align.mmd import mmd_rbf
from copy import deepcopy
from typing import Optional, Union
import torch
from torch import nn 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import tqdm
from methods.elasticdnn.model.vit import Linear_WrappedWithFBS

from utils.dl.common.model import get_model_device, get_model_size, set_module, get_module
import torch
from abc import abstractmethod


class ElasticDNN_OfflineFMModel_for_HuggingFaceFM(ElasticDNN_OfflineFMModel):
    def set_hugging_face_api(self, hugging_face_api: HuggingFaceModelAPI):
        self.hugging_face_api = hugging_face_api
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        return self.hugging_face_api.get_accuracy(self.models_dict['main'], test_loader, self.device, *args, **kwargs)
    
    def infer(self, x, *args, **kwargs):
        return self.hugging_face_api.infer(self.models_dict['main'], x, *args, **kwargs)
    
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        res = FM_to_MD_HuggingFaceFM_Util()
        res.set_hugging_face_api(self.hugging_face_api)
        return res.init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], reducing_width_ratio, samples)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return self.hugging_face_api.forward_to_get_task_loss(self.models_dict['main'], x, y)
    
    def get_feature_hook(self) -> LayerActivation:
        return self.hugging_face_api.get_feature_hook(self.models_dict['main'], self.device)
        
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        res = ElasticHuggingFaceFMUtil()
        res.set_hugging_face_api(self.hugging_face_api)
        return res
    
    def get_lora_util(self) -> FMLoRA_Util:
        res = FMLoRA_HuggingFaceFM_Util()
        res.set_hugging_face_api(self.hugging_face_api)
        return res
    
    def get_task_head_params(self):
        return self.hugging_face_api.get_task_head_params(self.models_dict['main'])
    
    
class ElasticDNN_OfflineMDModel_for_HuggingFaceFM(ElasticDNN_OfflineMDModel):
    def set_hugging_face_api(self, hugging_face_api: HuggingFaceModelAPI):
        self.hugging_face_api = hugging_face_api
        
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        return self.hugging_face_api.get_accuracy(self.models_dict['main'], test_loader, self.device, *args, **kwargs)
    
    def infer(self, x, *args, **kwargs):
        return self.hugging_face_api.infer(self.models_dict['main'], x, *args, **kwargs)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return self.hugging_face_api.forward_to_get_task_loss(self.models_dict['main'], x, y)
    
    def get_feature_hook(self) -> LayerActivation:
        return self.hugging_face_api.get_feature_hook(self.models_dict['main'], self.device)
    
    def get_distill_loss(self, student_output, teacher_output):
        return CrossEntropyLossSoft()(student_output, teacher_output)
    
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module):
        if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
            return None
        
        p = get_parameter(self.models_dict['main'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1:
            return None
        
        layers_name = self.hugging_face_api.get_qkv_proj_ff1_ff2_layer_names()
        if len(layers_name[0]) == 4:
            
            
            qkv_names = [layer[0] for layer in layers_name]
            qkv_proj_names = [layer[1] for layer in layers_name]
            ff1_names = [layer[-2] for layer in layers_name]
            ff2_names = [layer[-1] for layer in layers_name]
            
            qkv_weight_names = [n + '.weight' for n in qkv_names]
        
            if self_param_name in qkv_weight_names:
                ss = self_param_name.split('.')
                
                fm_qkv_name = '.'.join(ss[0: -1]) + '.qkv'
                fm_qkv = get_module(fm, fm_qkv_name)
                
                fm_abs_name = '.'.join(ss[0: -1]) + '.abs'
                fm_abs = get_module(fm, fm_abs_name)
                
                # print(fm_qkv_name, fm_abs_name, fm)
                
                return torch.cat([
                    fm_qkv.weight.data, # task-agnositc params
                    torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0) # task-specific params (LoRA)
                ], dim=0)
        else:
            q_names = [layer[0] for layer in layers_name]
            k_names = [layer[1] for layer in layers_name]
            v_names = [layer[2] for layer in layers_name]
            qkv_proj_names = [layer[3] for layer in layers_name]
            ff1_names = [layer[-2] for layer in layers_name]
            ff2_names = [layer[-1] for layer in layers_name]
            
            qkv_weight_names = [n + '.weight' for n in q_names + k_names + v_names]
        
            if self_param_name in qkv_weight_names:
            
                ss = self_param_name.split('.')
                # raise NotImplementedError() # TODO:
                fm_qkv_name = '.'.join(ss[0: -1]) + '.qkv'
                fm_qkv = get_module(fm, fm_qkv_name)
                
                fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
                fm_abs = get_module(fm, fm_abs_name)
                
                # print(fm_qkv_name, fm_abs_name, fm)
                
                return torch.cat([
                    fm_qkv.weight.data, # task-agnositc params
                    fm_abs[1].weight @ fm_abs[0].weight
                ], dim=0)
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
        
        ff1_weight_names = [n + '.linear.weight' for n in ff1_names]
        ff2_weight_names = [n + '.weight' for n in ff2_names]
            
        if self_param_name in ff1_weight_names:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)

        if self_param_name in ff2_weight_names:
            fm_param_name = self_param_name
            return get_parameter(fm, fm_param_name)
        
        return None
    
    
class ElasticHuggingFaceFMUtil(ElasticDNNUtil):
    def set_hugging_face_api(self, hugging_face_api: HuggingFaceModelAPI):
        self.hugging_face_api = hugging_face_api
        
    def convert_raw_dnn_to_master_dnn(self, raw_dnn: nn.Module, r: float, ignore_layers=[]):
        assert len(ignore_layers) == 0, 'not supported yet'

        raw_vit = deepcopy(raw_dnn)
        
        # set_module(module, 'patch_embed.proj', ProjConv_WrappedWithFBS(module.patch_embed.proj, r))
        layers = self.hugging_face_api.get_qkv_proj_ff1_ff2_layer_names()
        ff1_names = [layer[-2] for layer in layers]
                
        for name, module in raw_vit.named_modules():
            # if name.endswith('attn'):
            #     set_module(module, 'qkv', ToQKV_WrappedWithFBS(module.qkv, r))
            if name in ff1_names:
                # set_module(get_super_module(module, name), name.split('.')[-1], Linear_WrappedWithFBS(module, r))
                set_module(raw_vit, name, Linear_WrappedWithFBS(module, r))
        
        return raw_vit
    
    def set_master_dnn_sparsity(self, master_dnn: nn.Module, sparsity: float):
        return super().set_master_dnn_sparsity(master_dnn, sparsity)
    
    def select_most_rep_sample(self, master_dnn: nn.Module, samples: torch.Tensor):
        # print(samples)
        # return samples[0].unsqueeze(0)
        res = {k: v[0: 1] for k, v in samples.items()}
        return res
        
    def extract_surrogate_dnn_via_samples(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):
        sample = self.select_most_rep_sample(master_dnn, samples)
        # assert sample.dim() == 4 and sample.size(0) == 1
        
        # print('before')
        master_dnn.eval()
        self.clear_cached_channel_attention_in_master_dnn(master_dnn)
        with torch.no_grad():
            master_dnn_output = master_dnn(**sample)
            
        # print('after')
        
        boosted_vit = deepcopy(master_dnn)
        
        def get_unpruned_indexes_from_channel_attn(channel_attn: torch.Tensor, k):
            assert channel_attn.size(0) == 1, 'use A representative sample to generate channel attentions'
            
            res = channel_attn[0].nonzero(as_tuple=True)[0] # should be one-dim
            return res
        
        unpruned_indexes_of_layers = {}
        
        layers_name = self.hugging_face_api.get_qkv_proj_ff1_ff2_layer_names()
        ff1_names = [layer[-2] for layer in layers]
        ff2_names = [layer[-1] for layer in layers]
        
        for ff1_name, ff2_name in zip(ff1_names, ff2_names):
            ff_0 = get_module(boosted_vit, ff1_name)
            # ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, k)
            ff_0_pruned_indexes = ff_0.k_takes_all.cached_i[0].sort()[0]
            ff_0_unpruned_indexes = torch.LongTensor([ii for ii in range(ff_0.cached_channel_attention.size(1)) if ii not in ff_0_pruned_indexes])
            new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
            new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
            if ff_0.linear.bias is not None:
                new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
            # set_module(get_super_module(ff_0, ff1_name), ff1_name.split('.')[-1], 
            #            nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
            set_module(boosted_vit, ff1_name, 
                       nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
            
            ff_1 = get_module(boosted_vit, ff2_name)
            new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
            new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
            if ff_1.bias is not None:
                new_ff_1.bias.data.copy_(ff_1.bias.data)
            # set_module(get_super_module(ff_1), ff2_name.split('.')[-1],  new_ff_1)
            set_module(boosted_vit, ff2_name, new_ff_1)
            
            unpruned_indexes_of_layers[f'{ff1_name}.0.weight'] = ff_0_unpruned_indexes
        
        surrogate_dnn = boosted_vit
        surrogate_dnn.eval()
        surrogate_dnn = surrogate_dnn.to(get_model_device(master_dnn))
        # logger.debug(surrogate_dnn)
        with torch.no_grad():
            surrogate_dnn_output = surrogate_dnn(**sample)
            
        output_diff = ((surrogate_dnn_output - master_dnn_output) ** 2).sum()
        # assert output_diff < 1e-4, output_diff
        logger.info(f'output diff of master and surrogate DNN: {output_diff}')
        logger.debug(f'example output of master/surrogate: {master_dnn_output.sum(0)[0: 10]}, {surrogate_dnn_output.sum(0)[0: 10]}')
        # logger.info(f'\nonly prune mlp!!!!\n')
        # logger.info(f'\nonly prune mlp!!!!\n')
        
        if return_detail:
            return boosted_vit, unpruned_indexes_of_layers
        
        return boosted_vit
    
    def extract_surrogate_dnn_via_samples_with_perf_test(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):
        master_dnn_size = get_model_size(master_dnn, True)
        master_dnn_latency = self._get_model_latency(master_dnn, samples, 50, 
                                               get_model_device(master_dnn), 50, False)
        
        res = self.extract_surrogate_dnn_via_samples(master_dnn, samples, return_detail)
        if not return_detail:
            surrogate_dnn = res
        else:
            surrogate_dnn, unpruned_indexes_of_layers = res
        surrogate_dnn_size = get_model_size(surrogate_dnn, True)
        surrogate_dnn_latency = self._get_model_latency(master_dnn, samples, 50, 
                                               get_model_device(master_dnn), 50, False)

        logger.info(f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample) -> '
                    f'surrogate DNN ({surrogate_dnn_size:.3f}MB, {surrogate_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(master_dnn_size / surrogate_dnn_size):.2f}x, '
                    f'latency: ↓ {(master_dnn_latency / surrogate_dnn_latency):.2f}x)')
        
        return res
    
    def _get_model_latency(self, model: torch.nn.Module, model_input_size, sample_num: int, 
                           device: str, warmup_sample_num: int, return_detail=False):
        import time
        
        if isinstance(model_input_size, tuple):
            dummy_input = torch.rand(model_input_size).to(device)
        else:
            dummy_input = model_input_size
            
        model = model.to(device)
        model.eval()
        
        # warm up
        with torch.no_grad():
            for _ in range(warmup_sample_num):
                model(**dummy_input)
                
        infer_time_list = []
                
        if device == 'cuda' or 'cuda' in str(device):
            with torch.no_grad():
                for _ in range(sample_num):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    model(**dummy_input)
                    e.record()
                    torch.cuda.synchronize()
                    cur_model_infer_time = s.elapsed_time(e) / 1000.
                    infer_time_list += [cur_model_infer_time]

        else:
            with torch.no_grad():
                for _ in range(sample_num):
                    start = time.time()
                    model(**dummy_input)
                    cur_model_infer_time = time.time() - start
                    infer_time_list += [cur_model_infer_time]
                    
        avg_infer_time = sum(infer_time_list) / sample_num

        if return_detail:
            return avg_infer_time, infer_time_list
        return avg_infer_time
    
    
class FMLoRA_HuggingFaceFM_Util(FMLoRA_Util):
    def set_hugging_face_api(self, hugging_face_api: HuggingFaceModelAPI):
        self.hugging_face_api = hugging_face_api
    
    @torch.no_grad()
    def add_lora_ab_to_fm(self, fm: nn.Module, ab_r: int, samples: dict):
        fm.eval()
        
        if isinstance(samples, dict):
            o1 = fm(**samples)
        else:
            o1 = fm(samples)
        
        layers_name = self.hugging_face_api.get_qkv_proj_ff1_ff2_layer_names()
        if len(layers_name[0]) == 4:
            qkv_names = [layer[0] for layer in layers_name]
            
            from ..pipeline.offline.fm_lora.vit import ToQKV_WrappedWithLoRA
            for name, module in fm.named_modules():
                if name in qkv_names:
                    set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
        else:
            qkv_names = [layer[0] for layer in layers_name] + [layer[1] for layer in layers_name] + [layer[2] for layer in layers_name]
            
            from ..pipeline.offline.fm_lora.bert import ToQKV_WrappedWithLoRA
            for name, module in fm.named_modules():
                if name in qkv_names:
                    set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
        
        
        
        if isinstance(samples, dict):
            o2 = fm(**samples)
        else:
            o2 = fm(samples)
        
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
        if isinstance(samples, dict):
            o1 = fm(**samples)
        else:
            o1 = fm(samples)
        
        from ..pipeline.offline.fm_lora.vit import ToQKV_WrappedWithLoRA as ToQKV_WrappedWithLoRA1
        from ..pipeline.offline.fm_lora.bert import ToQKV_WrappedWithLoRA as ToQKV_WrappedWithLoRA2
        
        for name, module in fm.named_modules():
            if isinstance(module, ToQKV_WrappedWithLoRA1):
            
                qkv = module.qkv
                fm_abs = module.abs

                fm_abs_weight = torch.cat([_abs[1].weight @ _abs[0].weight for _abs in fm_abs], dim=0)
                qkv.weight.add_(fm_abs_weight)
                
                set_module(fm, name, qkv)
            
            elif isinstance(module, ToQKV_WrappedWithLoRA2):
            
                fc = module.fc
                ab = module.ab

                fc.weight.add_(ab[1].weight @ ab[0].weight)
                
                set_module(fm, name, fc)
        
        # print('absorb lora after')
        if isinstance(samples, dict):
            o2 = fm(**samples)
        else:
            o2 = fm(samples)
        
        if isinstance(o1, tuple):
            o1 = o1[-1]
            o2 = o2[-1]
        output_diff = ((o1 - o2) ** 2).sum()
        assert output_diff < 1e-6, output_diff
        
        return fm
        
        
class FM_to_MD_HuggingFaceFM_Util(FM_to_MD_Util):
    def set_hugging_face_api(self, hugging_face_api: HuggingFaceModelAPI):
        self.hugging_face_api = hugging_face_api
        
    def init_md_from_fm_by_reducing_width(self, fm: nn.Module, reducing_width_ratio: int) -> nn.Module:
        fm_vit = deepcopy(fm)
        
        # for block in fm_vit.bert.encoder.layer:
        #     set_module(block, 'attention.self', BertSelfAttentionPrunable.init_from_exist_self_attn(block.attention.self))
        
        def _f(n):
            return int(n // reducing_width_ratio)
        
        # def _rand_indexes(n):
            # return torch.randperm(n)[0: int(n // reducing_width_ratio)]
            
        def l1_max_indexes(p: torch.Tensor, dim=0):
            assert dim in [0, 1]
            assert p.dim() in [1, 2, 4]
            
            if dim == 1:
                p = p.T
            
            p_norm = p.abs().contiguous().view(p.size(0), -1).sum(dim=1)
            n = p.size(0)
            return p_norm.argsort(descending=True)[0: int(n // reducing_width_ratio)].sort()[0]
        
        layers_name = self.hugging_face_api.get_qkv_proj_ff1_ff2_layer_names()
        if len(layers_name[0]) == 6:
            q_names = [layer[0] for layer in layers_name]
            k_names = [layer[1] for layer in layers_name]
            v_names = [layer[2] for layer in layers_name]
            qkv_proj_names = [layer[3] for layer in layers_name]
            ff1_names = [layer[-2] for layer in layers_name]
            ff2_names = [layer[-1] for layer in layers_name]
            
            for q_name, k_name, v_name, qkv_proj_name, ff1_name, ff2_name in zip(q_names, k_names, v_names, qkv_proj_names, ff1_names, ff2_names):
                for k in [q_name, k_name, v_name]:
                    qkv = get_module(fm_vit, k)

                    new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                        qkv.bias is not None, qkv.weight.device)
                    indexes = l1_max_indexes(qkv.weight.data, 0)
                    
                    new_qkv.weight.data.copy_(qkv.weight.data[indexes])
                    if qkv.bias is not None:
                        new_qkv.bias.data.copy_(qkv.bias.data[indexes])
                    set_module(fm_vit, k, new_qkv)
                    
                proj = get_module(fm_vit, qkv_proj_name)
                new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
                                    proj.bias is not None, proj.weight.device)
                new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
                if proj.bias is not None:
                    new_proj.bias.data.copy_(proj.bias.data)
                set_module(fm_vit, qkv_proj_name, new_proj)
                
                fc1 = get_module(fm_vit, ff1_name)
                new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
                                    fc1.bias is not None, fc1.weight.device)
                indexes = l1_max_indexes(fc1.weight.data, 0)
                new_fc1.weight.data.copy_(fc1.weight.data[indexes])
                if fc1.bias is not None:
                    new_fc1.bias.data.copy_(fc1.bias.data[indexes])
                set_module(fm_vit, ff1_name, new_fc1)

                fc2 = get_module(fm_vit, ff2_name)
                new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
                                    fc2.bias is not None, fc2.weight.device)
                new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
                if fc2.bias is not None:
                    new_fc2.bias.data.copy_(fc2.bias.data)
                set_module(fm_vit, ff2_name, new_fc2)
                
        if len(layers_name[0]) == 4:
            qkv_names = [layer[0] for layer in layers_name]
            qkv_proj_names = [layer[1] for layer in layers_name]
            ff1_names = [layer[-2] for layer in layers_name]
            ff2_names = [layer[-1] for layer in layers_name]
            
            for qkv_name, qkv_proj_name, ff1_name, ff2_name in zip(qkv_names, qkv_proj_names, ff1_names, ff2_names):
                qkv = get_module(fm_vit, qkv_name)
                new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                    qkv.bias is not None, qkv.weight.device)
                indexes = l1_max_indexes(qkv.weight.data, 0)
                
                new_qkv.weight.data.copy_(qkv.weight.data[indexes])
                if qkv.bias is not None:
                    new_qkv.bias.data.copy_(qkv.bias.data[indexes])
                set_module(fm_vit, qkv_name, new_qkv)
                    
                proj = get_module(fm_vit, qkv_proj_name)
                new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
                                    proj.bias is not None, proj.weight.device)
                new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
                if proj.bias is not None:
                    new_proj.bias.data.copy_(proj.bias.data)
                set_module(fm_vit, qkv_proj_name, new_proj)
                
                fc1 = get_module(fm_vit, ff1_name)
                new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
                                    fc1.bias is not None, fc1.weight.device)
                indexes = l1_max_indexes(fc1.weight.data, 0)
                new_fc1.weight.data.copy_(fc1.weight.data[indexes])
                if fc1.bias is not None:
                    new_fc1.bias.data.copy_(fc1.bias.data[indexes])
                set_module(fm_vit, ff1_name, new_fc1)

                fc2 = get_module(fm_vit, ff2_name)
                new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
                                    fc2.bias is not None, fc2.weight.device)
                new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
                if fc2.bias is not None:
                    new_fc2.bias.data.copy_(fc2.bias.data)
                set_module(fm_vit, ff2_name, new_fc2)
            
        return fm_vit
    
    def init_md_from_fm_by_reducing_width_with_perf_test(self, fm: nn.Module, reducing_width_ratio: int,
                                                         samples: torch.Tensor) -> nn.Module:
        fm_size = get_model_size(fm, True)
        fm_latency = self._get_model_latency(fm, samples, 20, 
                                               get_model_device(fm), 20, False)
        
        master_dnn = self.init_md_from_fm_by_reducing_width(fm, reducing_width_ratio)
        master_dnn_size = get_model_size(master_dnn, True)
        logger.debug(f'inited master DNN: {master_dnn}')
        master_dnn_latency = self._get_model_latency(master_dnn, samples, 20, 
                                               get_model_device(master_dnn), 20, False)

        logger.info(f'init master DNN (w/o FBS yet) by reducing foundation model\'s width (by {reducing_width_ratio:d}x)')
        logger.info(f'foundation model ({fm_size:.3f}MB, {fm_latency:.4f}s/sample) -> '
                    f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(fm_size / master_dnn_size):.2f}x, '
                    f'latency: ↓ {(fm_latency / master_dnn_latency):.2f}x)')
        
        return master_dnn
    
    def _get_model_latency(self, model: torch.nn.Module, model_input_size, sample_num: int, 
                           device: str, warmup_sample_num: int, return_detail=False):
        import time
        
        if isinstance(model_input_size, tuple):
            dummy_input = torch.rand(model_input_size).to(device)
        else:
            dummy_input = model_input_size
            
        model = model.to(device)
        model.eval()
        
        # warm up
        with torch.no_grad():
            for _ in range(warmup_sample_num):
                if isinstance(dummy_input, dict):
                    model(**dummy_input)
                else:
                    model(dummy_input)
                
        infer_time_list = []
                
        if device == 'cuda' or 'cuda' in str(device):
            with torch.no_grad():
                for _ in range(sample_num):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    
                    if isinstance(dummy_input, dict):
                        model(**dummy_input)
                    else:
                        model(dummy_input)
                    
                    e.record()
                    torch.cuda.synchronize()
                    cur_model_infer_time = s.elapsed_time(e) / 1000.
                    infer_time_list += [cur_model_infer_time]

        else:
            with torch.no_grad():
                for _ in range(sample_num):
                    start = time.time()
                    
                    if isinstance(dummy_input, dict):
                        model(**dummy_input)
                    else:
                        model(dummy_input)
                    
                    cur_model_infer_time = time.time() - start
                    infer_time_list += [cur_model_infer_time]
                    
        avg_infer_time = sum(infer_time_list) / sample_num

        if return_detail:
            return avg_infer_time, infer_time_list
        return avg_infer_time