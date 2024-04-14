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
app_name = 'det'
sd_sparsity = 0.6

settings = {
    'involve_fm': True
}

scenario = build_scenario(
    source_datasets_name=['GTA5Det', 'SuperviselyPersonDet'],
    target_datasets_order=['CityscapesDet', 'BaiduPersonDet'] * 15,
    da_mode='close_set',
    data_dirs={
        'GTA5Det': '/data/zql/datasets/GTA-ls-copy/GTA5',
        'SuperviselyPersonDet': '/data/zql/datasets/supervisely_person_full_20230635/Supervisely Person Dataset',
        'CityscapesDet': '/data/zql/datasets/cityscape/',
        'BaiduPersonDet': '/data/zql/datasets/baidu_person/clean_images/'
    },
)

from experiments.elasticdnn.vit_b_16.online_new.det.model import ElasticDNN_DetOnlineModel
elasticfm_model = ElasticDNN_DetOnlineModel('cls', init_online_model(
    'experiments/elasticdnn/vit_b_16/offline/fm_to_md/det/results/det_md_w_fbs_index.py/20230703/999994-214617-trial_in_card1/models/fm_best.pt',
    'experiments/elasticdnn/vit_b_16/offline/fm_to_md/det/results/det_md_w_fbs_index.py/20230703/999994-214617-trial_in_card1/models/md_best.pt',
    'det', __file__
), device, {
    'md_to_fm_alpha': 1.0,
    'fm_to_md_alpha': 1.0
}, scenario.num_classes)

da_alg = FeatAlignAlg
from experiments.elasticdnn.vit_b_16.online_new.det.model import DetOnlineFeatAlignModel
da_model = DetOnlineFeatAlignModel
da_alg_hyp = {'CityscapesDet': {
    'train_batch_size': 8,
    'val_batch_size': 32,
    'num_workers': 16,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 1e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 100,
    'val_freq': 20,
    'sd_sparsity': 0.6,
    'feat_align_loss_weight': 0.3
}, 'BaiduPersonDet': {
    'train_batch_size': 8,
    'val_batch_size': 32,
    'num_workers': 16,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 1e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 100,
    'val_freq': 20,
    'sd_sparsity': 0.6,
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
