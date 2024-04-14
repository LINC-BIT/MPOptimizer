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
from sam import FM_to_MD_sam_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from sam import FMLoRA_sam_Util
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
from new_impl.cv.utils.baseline_da import baseline_da

device = 'cuda'
app_name = 'cls'

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


da_alg = FeatAlignAlg
#from experiments.cua.vit_b_16.online.cls.model import ClsOnlineFeatAlignModel
da_model = SegOnlineFeatAlignModel(
    app_name,
    'new_impl/cv/sam/results/seg_wo_fbs.py/20231130/999999-144157/models/md_best.pt',
    device,
    scenario.num_classes
)
da_alg_hyp = {'Cityscapes': {
    'train_batch_size': 16,
    'val_batch_size': 128,
    'num_workers': 16,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 1e-9, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 10,
    'val_freq': 20,
    # 'sd_sparsity': 0.8,
    'feat_align_loss_weight': 3.0
}, 'BaiduPerson': {
    'train_batch_size': 16,
    'val_batch_size': 128,
    'num_workers': 16,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 1e-2, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 10,
    'val_freq': 20,
    # 'sd_sparsity': 0.8,
    'feat_align_loss_weight': 0.3
}}


baseline_da(
    app_name,
    scenario,
    da_alg,
    da_alg_hyp,
    da_model,
    device,
    __file__,
    sys.argv[0]
)
