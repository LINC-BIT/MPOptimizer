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
from methods.feat_align.main import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from methods.feat_align.mmd import mmd_rbf
from experiments.utils.elasticfm_da import init_online_model, elasticfm_da

device = 'cuda'
app_name = 'cls'
sd_sparsity = 0.8

settings = {
    'involve_fm': True
}

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

from experiments.elasticdnn.vit_b_16.online_new.cls.model import ElasticDNN_ClsOnlineModel
elasticfm_model = ElasticDNN_ClsOnlineModel('cls', init_online_model(
    # 'experiments/elasticdnn/vit_b_16/offline/fm_to_md/results/cls_md_index.py/20230529/star_999997-154037-only_prune_mlp/models/fm_best.pt',
    # 'experiments/elasticdnn/vit_b_16/offline/fm_to_md/results/cls_md_index.py/20230529/star_999997-154037-only_prune_mlp/models/md_best.pt',
    'experiments/elasticdnn/vit_b_16/offline/fm_to_md/cls/results/cls_md_index.py/20230617/999992-101343-lr1e-5_index_bug_fixed/models/fm_best.pt',
    'experiments/elasticdnn/vit_b_16/offline/fm_to_md/cls/results/cls_md_index.py/20230617/999992-101343-lr1e-5_index_bug_fixed/models/md_best.pt',
    'cls', __file__
), device, {
    'md_to_fm_alpha': 1,
    'fm_to_md_alpha': 1
})

da_alg = FeatAlignAlg
from experiments.elasticdnn.vit_b_16.online_new.cls.model import ClsOnlineFeatAlignModel
da_model = ClsOnlineFeatAlignModel
da_alg_hyp = {
    'CityscapesCls': {
        'train_batch_size': 64,
        'val_batch_size': 512,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 4e-6/2, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'sd_sparsity': sd_sparsity,
        'feat_align_loss_weight': 3.0
    },
    'BaiduPersonCls': {
        'train_batch_size': 64,
        'val_batch_size': 512,
        'num_workers': 8,
        'optimizer': 'SGD',
        'optimizer_args': {'lr': 1e-4/2, 'momentum': 0.9},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'sd_sparsity': sd_sparsity,
        'feat_align_loss_weight': 0.3
    }
}


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
