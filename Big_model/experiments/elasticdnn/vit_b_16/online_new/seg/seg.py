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
app_name = 'seg'
sd_sparsity = 0.8

settings = {
    'involve_fm': True
}

scenario = build_scenario(
    source_datasets_name=['GTA5', 'SuperviselyPerson'],
    target_datasets_order=['Cityscapes', 'BaiduPerson'] * 15,
    da_mode='close_set',
    data_dirs={
        'GTA5': '/data/zql/datasets/GTA-ls-copy/GTA5',
        'SuperviselyPerson': '/data/zql/datasets/supervisely_person/Supervisely Person Dataset',
        'Cityscapes': '/data/zql/datasets/cityscape/',
        'BaiduPerson': '/data/zql/datasets/baidu_person/clean_images/'
    },
)

from experiments.elasticdnn.vit_b_16.online_new.seg.model import ElasticDNN_SegOnlineModel
elasticfm_model = ElasticDNN_SegOnlineModel('cls', init_online_model(
    # 'experiments/elasticdnn/vit_b_16/offline/fm_to_md/results/cls_md_index.py/20230529/star_999997-154037-only_prune_mlp/models/fm_best.pt',
    # 'experiments/elasticdnn/vit_b_16/offline/fm_to_md/results/cls_md_index.py/20230529/star_999997-154037-only_prune_mlp/models/md_best.pt',
    'experiments/elasticdnn/vit_b_16/offline/fm_to_md/seg/results/seg_md_index.py/20230613/999995-093643-new_md_structure/models/fm_best.pt',
    'experiments/elasticdnn/vit_b_16/offline/fm_to_md/seg/results/seg_md_index.py/20230613/999995-093643-new_md_structure/models/md_best.pt',
    'seg', __file__
), device, {
    'md_to_fm_alpha': 0.1,
    'fm_to_md_alpha': 0.1
}, scenario.num_classes)

da_alg = FeatAlignAlg
from experiments.elasticdnn.vit_b_16.online_new.seg.model import SegOnlineFeatAlignModel
da_model = SegOnlineFeatAlignModel
da_alg_hyp = {'Cityscapes': {
    'train_batch_size': 16,
    'val_batch_size': 128,
    'num_workers': 16,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 3e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 100,
    'val_freq': 20,
    'sd_sparsity': 0.8,
    'feat_align_loss_weight': 0.3
}, 'BaiduPerson': {
    'train_batch_size': 16,
    'val_batch_size': 128,
    'num_workers': 16,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 1e-7, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 100,
    'val_freq': 20,
    'sd_sparsity': 0.8,
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
    sys.argv[1]
)
