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
from methods.gem.gem_el_bert import GEMAlg
import tqdm
from methods.feat_align.mmd import mmd_rbf
from experiments.utils.elasticfm_da import init_online_model, elasticfm_da

device = 'cuda'
app_name = 'secls'
sd_sparsity = 0.8

settings = {
    'involve_fm': True
}

scenario = build_scenario(
    source_datasets_name=['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                            'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610'],
    target_datasets_order=['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker', 
                           'Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
                           'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
                           'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100', 
                           'SemEval-Laptop', 'SemEval-Rest'] * 2 + ['Liu3Domains-Computer', 'Liu3Domains-Router'],
    da_mode='close_set',
    data_dirs={
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
            for k in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                            'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing3Domains/asc/{k.split("-")[1]}' 
            for k in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker']},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing9Domains/asc/{k.split("-")[1]}' 
            for k in ['Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
                           'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
                           'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100']},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/XuSemEval/asc/14/{k.split("-")[1].lower()}' 
            for k in ['SemEval-Laptop', 'SemEval-Rest']},
    },
)

from experiments.elasticdnn.bert_base.online.se_cls_cl.model import ElasticDNN_SeClsOnlineModel
elasticfm_model = ElasticDNN_SeClsOnlineModel('secls', init_online_model(
    'experiments/elasticdnn/bert_base/offline/fm_to_md/se_cls/results/secls_md_w_fbs_index.py/20230704/999994-085209-logic_verify/models/fm_best.pt',
    'experiments/elasticdnn/bert_base/offline/fm_to_md/se_cls/results/secls_md_w_fbs_index.py/20230704/999994-085209-logic_verify/models/md_best.pt',
    'cls', __file__
), device, {
    'md_to_fm_alpha': 0.2,
    'fm_to_md_alpha': 0.2
})

da_alg = GEMAlg
from experiments.elasticdnn.bert_base.online.se_cls_cl.model import SeClsOnlineGEMModel
da_model = SeClsOnlineGEMModel
da_alg_hyp = {
    'train_batch_size': 16,
    'val_batch_size': 64,
    'num_workers': 8,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 100,
    'val_freq': 20,
    'n_memories': 16,
    'n_inputs': 3 * 224 * 224,
    'margin': 0.5,
    'num_my_iters': 0,
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
