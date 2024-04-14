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
from methods.feat_align.main import FeatAlignAlg
import tqdm
from methods.feat_align.mmd import mmd_rbf
from experiments.utils.elasticfm_da import init_online_model, elasticfm_da


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

device = 'cuda'
app_name = 'vqa'
sd_sparsity = 0.8

settings = {
    'involve_fm': True
}

target_datasets = ['VQAv2_split1_c_gaussian_noise', 'VQAv2_split1_c_shot_noise', 'VQAv2_split1_c_impulse_noise', 'VQAv2_split1_c_defocus_blur', 'VQAv2_split1_c_glass_blur', 'VQAv2_split1_c_motion_blur', 'VQAv2_split1_c_zoom_blur', 'VQAv2_split1_c_snow', 'VQAv2_split1_c_frost', 'VQAv2_split1_c_fog', 'VQAv2_split1_c_brightness', 'VQAv2_split1_c_contrast', 'VQAv2_split1_c_elastic_transform', 'VQAv2_split1_c_pixelate', 'VQAv2_split1_c_jpeg_compression', 'VQAv2_split1_c_speckle_noise', 'VQAv2_split1_c_gaussian_blur', 'VQAv2_split1_c_spatter', 'VQAv2_split1_c_saturate'] * 2
target_datasets = target_datasets[0: 30]
assert len(target_datasets) == 30

scenario = build_scenario(
    source_datasets_name=['VQAv2_split1'],
    target_datasets_order=target_datasets,
    da_mode='close_set',
    data_dirs={
        k: '/data/zql/datasets/vqav2' for k in ['VQAv2_split1', 'VQAv2_split1_c_gaussian_noise', 'VQAv2_split1_c_shot_noise', 'VQAv2_split1_c_impulse_noise', 'VQAv2_split1_c_defocus_blur', 'VQAv2_split1_c_glass_blur', 'VQAv2_split1_c_motion_blur', 'VQAv2_split1_c_zoom_blur', 'VQAv2_split1_c_snow', 'VQAv2_split1_c_frost', 'VQAv2_split1_c_fog', 'VQAv2_split1_c_brightness', 'VQAv2_split1_c_contrast', 'VQAv2_split1_c_elastic_transform', 'VQAv2_split1_c_pixelate', 'VQAv2_split1_c_jpeg_compression', 'VQAv2_split1_c_speckle_noise', 'VQAv2_split1_c_gaussian_blur', 'VQAv2_split1_c_spatter', 'VQAv2_split1_c_saturate']
    },
)

from experiments.elasticdnn.vilt.online.vqa.model import ElasticDNN_VQAOnlineModel
elasticfm_model = ElasticDNN_VQAOnlineModel('vqa', init_online_model(
    'experiments/elasticdnn/vilt/offline/fm_to_md/vqa/results/vqa_w_fbs_index.py/20230731/999999-095720-trial/models/fm_best.pt',
    'experiments/elasticdnn/vilt/offline/fm_to_md/vqa/results/vqa_w_fbs_index.py/20230731/999999-095720-trial/models/md_best.pt',
    'vqa', __file__
), device, {
    'md_to_fm_alpha': 0.2,
    'fm_to_md_alpha': 0.2
})

da_alg = FeatAlignAlg
from experiments.elasticdnn.vilt.online.vqa.model import VQAOnlineFeatAlignModel
da_model = VQAOnlineFeatAlignModel
da_alg_hyp = {
    'train_batch_size': 64,
    'val_batch_size': 256,
    'num_workers': 0,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 1e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 100,
    'val_freq': 20,
    'feat_align_loss_weight': 1.0,
    'sd_sparsity': 0.7
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
    sys.argv[1]
)
