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
from clip import FM_to_MD_clip_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from clip import FMLoRA_clip_Util
from clip import ElasticclipUtil
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
    source_datasets_name=['GTA5Cls', 'SuperviselyPersonCls'],
    target_datasets_order=['CityscapesCls', 'BaiduPersonCls'] * 15,
    da_mode='close_set',
    data_dirs={
        'GTA5Cls': '/data/zql/datasets/gta5_for_cls_task',
        'SuperviselyPersonCls': '/data/zql/datasets/supervisely_person_for_cls_task',
        'CityscapesCls': '/data/zql/datasets/cityscapes_for_cls_task',
        'BaiduPersonCls': '/data/zql/datasets/baiduperson_for_cls_task'
    },
)
class ClsOnlineFeatAlignModel(OnlineFeatAlignModel):
    def get_trained_params(self): # TODO: elastic fm only train a part of params
        #qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters() if 'attention.attention.projection_query' in n or 'attention.attention.projection_key' in n or 'attention.attention.projection_value' in n or 'intermediate.dense' in n or 'output.dense' in n]
        qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters()]
        return qkv_and_norm_params
    
    def get_feature_hook(self):
        return LayerActivation(get_module(self.models_dict['main'], 'classifier'), False, self.device)
    
    def forward_to_get_task_loss(self, x, y):
        return F.cross_entropy(self.infer(x), y)
    
    def get_mmd_loss(self, f1, f2):
        return mmd_rbf(f1, f2)
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](x)
    
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


da_alg = FeatAlignAlg
#from experiments.cua.vit_b_16.online.cls.model import ClsOnlineFeatAlignModel
da_model = ClsOnlineFeatAlignModel(
    app_name,
    'new_impl/cv/clip/results/cls_md_wo_fbs.py/20231115/999998-195939-/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/clip/cls_md_wo_fbs.py/models/md_best.pt',
    device
)
da_alg_hyp = {
    'CityscapesCls': {
        'train_batch_size': 64,
        'val_batch_size': 512,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 4e-8/2, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'feat_align_loss_weight': 3.0
    },
    'BaiduPersonCls': {
        'train_batch_size': 64,
        'val_batch_size': 512,
        'num_workers': 8,
        'optimizer': 'SGD',
        'optimizer_args': {'lr': 1e-10, 'momentum': 0.9},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'feat_align_loss_weight': 0.2
    }
}


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
