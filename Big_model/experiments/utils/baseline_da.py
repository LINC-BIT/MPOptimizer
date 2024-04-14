import sys
from utils.dl.common.env import set_random_seed
set_random_seed(1)

from typing import List
from data.dataloader import build_dataloader
from data import Scenario
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
import shutil
from utils.common.log import logger
from utils.common.data_record import write_json
# from methods.shot.shot import OnlineShotModel
from methods.feat_align.main import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from methods.feat_align.mmd import mmd_rbf
from methods.base.alg import BaseAlg
from methods.base.model import BaseModel


def baseline_da(app_name: str,
                 scenario: Scenario, 
                 da_alg: BaseAlg, 
                 da_alg_hyp: dict,
                 da_model: BaseModel,
                 device,
                 __entry_file__,
                 tag=None):
    
    # involve_fm = settings['involve_fm']
    
    task_name = app_name
    # online_model = elasticfm_model
    
    log_dir = get_res_save_dir(__entry_file__, tag=tag)
    tb_writer = create_tbwriter(os.path.join(log_dir, 'tb_log'), False)
    res = []
    global_avg_after_acc = 0.
    global_iter = 0
    
    for domain_index, _ in enumerate(scenario.target_domains_order):
        
        cur_target_domain_name = scenario.target_domains_order[scenario.cur_domain_index]
        if cur_target_domain_name in da_alg_hyp:
            da_alg_hyp = da_alg_hyp[cur_target_domain_name]
            logger.info(f'use dataset-specific hyps')
        
        # tmp_sd_path = os.path.join(log_dir, 'tmp_sd_model.pt')
        # torch.save({'main': sd}, tmp_sd_path)
        
        # if task_name != 'cls':
        #     da_model_args = [f'{task_name}/{domain_index}', 
        #             tmp_sd_path, 
        #             device,
        #             scenario.num_classes]
        # else:
        #     da_model_args = [f'{task_name}/{domain_index}', 
        #             tmp_sd_path, 
        #             device]
        # cur_da_model = da_model(*da_model_args)
        
        da_metrics, after_da_model = da_alg(
            {'main': da_model}, 
            os.path.join(log_dir, f'{task_name}/{domain_index}')
        ).run(scenario, da_alg_hyp)
        # os.remove(tmp_sd_path)
        
        if domain_index > 0:
            shutil.rmtree(os.path.join(log_dir, f'{task_name}/{domain_index}/backup_codes'))
        
        accs = da_metrics['accs']
        before_acc = accs[0]['acc']
        after_acc = accs[-1]['acc']
        
        tb_writer.add_scalars(f'accs/{task_name}', dict(before=before_acc, after=after_acc), domain_index)
        tb_writer.add_scalar(f'times/{task_name}', da_metrics['time'], domain_index)
        
        for _acc in accs:
            tb_writer.add_scalar('total_acc', _acc['acc'], _acc['iter'] + global_iter)
        global_iter += _acc['iter'] + 1
        
        scenario.next_domain()
        
        logger.info(f"task: {task_name}, domain {domain_index}, acc: {before_acc:.4f} -> "
                    f"{after_acc:.4f} ({da_metrics['time']:.2f}s)")
        
        global_avg_after_acc += after_acc
        cur_res = da_metrics
        res += [cur_res]
        write_json(os.path.join(log_dir, 'res.json'), res, backup=False)

    global_avg_after_acc /= (domain_index + 1)
    logger.info(f'-----> final metric: {global_avg_after_acc:.4f}')
    write_json(os.path.join(log_dir, f'res_{global_avg_after_acc:.4f}.json'), res, backup=False)
    