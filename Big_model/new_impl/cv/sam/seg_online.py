from typing import List
from data.dataloader import build_dataloader
# from methods.elasticdnn.api.online_model import ElasticDNN_OnlineModel
from new_impl.cv.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel

import torch
import sys
from torch import nn
from new_impl.cv.elasticdnn.api.model import ElasticDNN_OfflineSegFMModel, ElasticDNN_OfflineSegMDModel
from new_impl.cv.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from new_impl.cv.elasticdnn.model.base import ElasticDNNUtil
from new_impl.cv.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.vit import FMLoRA_ViT_Util
from sam import ElasticsamUtil
from utils.common.file import ensure_dir
from utils.dl.common.model import LayerActivation, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F
from utils.dl.common.env import create_tbwriter
import os
from utils.common.log import logger
from utils.common.data_record import write_json
# from methods.shot.shot import OnlineShotModel
from new_impl.cv.feat_align.main import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from new_impl.cv.feat_align.mmd import mmd_rbf
from new_impl.cv.utils.elasticfm_da import init_online_model, elasticfm_da
torch.cuda.set_device(1)
device = 'cuda'
app_name = 'seg'
sd_sparsity = 0.

settings = {
    'involve_fm': True
}

scenario = build_scenario(
    source_datasets_name=['GTA5', 'SuperviselyPerson'],
    target_datasets_order=['Cityscapes', 'BaiduPerson'] * 10,
    da_mode='close_set',
    data_dirs={
        'GTA5': '/data/zql/datasets/GTA-ls-copy/GTA5',
        'SuperviselyPerson': '/data/zql/datasets/supervisely_person/Supervisely Person Dataset',
        'Cityscapes': '/data/zql/datasets/cityscape/',
        'BaiduPerson': '/data/zql/datasets/baidu_person/clean_images/'
    },
)


class ElasticDNN_SegOnlineModel(ElasticDNN_OnlineModel):
    def __init__(self, name: str, models_dict_path: str, device: str, ab_options: dict, num_classes: int):
        super().__init__(name, models_dict_path, device, ab_options)
        self.num_classes = num_classes
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        device = self.device
        self.to_eval_mode()
        from methods.elasticdnn.api.model import StreamSegMetrics
        metrics = StreamSegMetrics(self.num_classes)
        metrics.reset()
        import tqdm
        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch_index, (x, y) in pbar:
                x, y = x.to(device, dtype=x.dtype, non_blocking=True, copy=False), \
                    y.to(device, dtype=y.dtype, non_blocking=True, copy=False)
                output = self.infer(x)
                pred = output.detach().max(dim=1)[1].cpu().numpy()
                metrics.update((y + 0).cpu().numpy(), pred)
                
                res = metrics.get_results()
                pbar.set_description(f'cur batch mIoU: {res["Mean Acc"]:.4f}')
                
        res = metrics.get_results()
        return res['Mean Acc']
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticsamUtil()
    
    def get_fm_matched_param_of_md_param(self, md_param_name):
        # only between qkv.weight, norm.weight/bias
        self_param_name = md_param_name
        fm = self.models_dict['fm']
        if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
            return None
        
        p = get_parameter(self.models_dict['md'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1 and 'layernorm' in self_param_name and 'weight' in self_param_name:
            if self_param_name.startswith('norm'):
                return None
            return get_parameter(fm, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if 'qkv.weight' in self_param_name:
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            # NOTE: unrecoverable operation! multiply LoRA parameters to allow it being updated in update_fm_param()
            # TODO: if fm will be used for inference, _mul_lora_weight will not be applied!
            if not hasattr(fm_abs, '_mul_lora_weight'):
                logger.debug(f'set _mul_lora_weight in {fm_abs_name}')
                setattr(fm_abs, '_mul_lora_weight', 
                        nn.Parameter(fm_abs[1].weight @ fm_abs[0].weight))
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs._mul_lora_weight.data # task-specific params (LoRA)
            ], dim=0)
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
            
        elif 'mlp.lin1' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)

        elif 'mlp.lin2' in self_param_name and 'weight' in self_param_name:
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
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            fm_qkv.weight.data.copy_(new_fm_attn_weight)
            
            # update task-specific parameters
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
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
        elif p.dim() == 1 and 'layernorm' in self_param_name and 'weight' in self_param_name:
            return get_parameter(md, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if 'qkv.weight' in self_param_name:
            return get_parameter(md, self_param_name)
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
            
        elif 'mlp.lin1.0.weight' in self_param_name:
            fm_param_name = '.'.join(self_param_name.split('.')[0: -2]) + '.linear.weight'
            return get_parameter(md, fm_param_name)

        elif 'mlp.lin2' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name
            return get_parameter(md, fm_param_name)
        
        else:
            # return get_parameter(fm, self_param_name)
            return None
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['sd'], 'head')
        return list(head.parameters())
    

class SegOnlineFeatAlignModel(OnlineFeatAlignModel):
    def __init__(self, name: str, models_dict_path: str, device: str, num_classes):
        super().__init__(name, models_dict_path, device)
        self.num_classes = num_classes
        
    def get_feature_hook(self):
        return LayerActivation(get_module(self.models_dict['main'], 'head'), False, self.device)
    
    def forward_to_get_task_loss(self, x, y):
        return F.cross_entropy(self.infer(x), y)
    
    def get_mmd_loss(self, f1, f2):
        return mmd_rbf(f1.flatten(1), f2.flatten(1))
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](x)
    
    def get_trained_params(self):
        qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters() if 'qkv.weight' in n or 'norm' in n or 'mlp' in n]
        return qkv_and_norm_params
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](x)
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        device = self.device
        self.to_eval_mode()
        from methods.elasticdnn.api.model import StreamSegMetrics
        metrics = StreamSegMetrics(self.num_classes)
        metrics.reset()
        import tqdm
        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch_index, (x, y) in pbar:
                x, y = x.to(device, dtype=x.dtype, non_blocking=True, copy=False), \
                    y.to(device, dtype=y.dtype, non_blocking=True, copy=False)
                output = self.infer(x)
                pred = output.detach().max(dim=1)[1].cpu().numpy()
                metrics.update((y + 0).cpu().numpy(), pred)
                
                res = metrics.get_results()
                pbar.set_description(f'cur batch mIoU: {res["Mean Acc"]:.4f}')
                
        res = metrics.get_results()
        return res['Mean Acc']
    
    




#from new_impl.cv.model import ElasticDNN_ClsOnlineModel
elasticfm_model = ElasticDNN_SegOnlineModel('cls', init_online_model(
    # 'experiments/elasticdnn/vit_b_16/offline/fm_to_md/results/cls_md_index.py/20230529/star_999997-154037-only_prune_mlp/models/fm_best.pt',
    # 'experiments/elasticdnn/vit_b_16/offline/fm_to_md/results/cls_md_index.py/20230529/star_999997-154037-only_prune_mlp/models/md_best.pt',
    #'experiments/elasticdnn/vit_b_16/offline/fm_to_md/cls/results/cls_md_index.py/20230617/999992-101343-lr1e-5_index_bug_fixed/models/fm_best.pt',
    #'experiments/elasticdnn/vit_b_16/offline/fm_to_md/cls/results/cls_md_index.py/20230617/999992-101343-lr1e-5_index_bug_fixed/models/md_best.pt',
    'new_impl/cv/sam/results/seg_wo_index.py/20231125/999999-175801-/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/sam/seg_wo_index.py/models/fm_best.pt',
    'new_impl/cv/sam/results/seg_wo_index.py/20231125/999999-175801-/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/sam/seg_wo_index.py/models/md_best.pt',
    'seg', __file__
), device, {
    'md_to_fm_alpha': 0.1,
    'fm_to_md_alpha': 0.1
},scenario.num_classes)

da_alg = FeatAlignAlg
from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
#from new_impl.cv.model import ClsOnlineFeatAlignModel
da_model = SegOnlineFeatAlignModel
da_alg_hyp = {'Cityscapes': {
    'train_batch_size': 16,
    'val_batch_size': 128,
    'num_workers': 16,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 3e-5, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 100,
    'val_freq': 20,
    'sd_sparsity': 0.5,
    'feat_align_loss_weight': 0.3
}, 'BaiduPerson': {
    'train_batch_size': 16,
    'val_batch_size': 128,
    'num_workers': 16,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 1e-7,'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 100,
    'val_freq': 20,
    'sd_sparsity': 0.5,
    'feat_align_loss_weight': 0.3
}}


elasticfm_da(
    [app_name],
    [scenario],
    [elasticfm_model],
    [da_alg],
    [da_alg_hyp],
    [da_model],
    device,
    settings,
    __file__,
    sys.argv[0]
)
