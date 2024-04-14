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
from new_impl.cv.elasticdnn.model.vit import ElasticViTUtil
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
from new_impl.cv.feat_align.main import FeatAlignAlg
import tqdm
from new_impl.cv.feat_align.mmd import mmd_rbf
from new_impl.cv.utils.elasticfm_da import init_online_model, elasticfm_da


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.cuda.set_device(1)
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
    source_datasets_name=['VQA_split1'],
    target_datasets_order=['VQA_split1_c'],
    da_mode='close_set',
    data_dirs={
        k: '/data/zql/datasets/vqav2' for k in ['VQA_split1', 'VQA_split1_c']
    },
)

from blip import ElasticDNN_VQAOnlineModel
elasticfm_model = ElasticDNN_VQAOnlineModel('vqa', init_online_model(
    '',
    '',
    'vqa', __file__
), device, {
    'md_to_fm_alpha': 0.2,
    'fm_to_md_alpha': 0.2
})

da_alg = FeatAlignAlg
from blip import VQAOnlineFeatAlignModel
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
    sys.argv[0]
)
