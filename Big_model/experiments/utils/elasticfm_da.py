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
from utils.common.log import logger
from utils.common.data_record import write_json
# from methods.shot.shot import OnlineShotModel
from methods.feat_align.main import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from methods.feat_align.mmd import mmd_rbf
from methods.base.alg import BaseAlg
import shutil
from methods.base.model import BaseModel


def elasticfm_da(apps_name: List[str],
                 scenarios: List[Scenario], 
                 elasticfm_models: List[ElasticDNN_OnlineModel], 
                 da_algs: List[BaseAlg], 
                 da_alg_hyps: List[dict],
                 da_models: List[BaseModel],
                 device,
                 settings,
                 __entry_file__,
                 tag=None):
    
    involve_fm = settings['involve_fm']
    
    tasks_name = apps_name
    online_models = elasticfm_models
    
    log_dir = get_res_save_dir(__entry_file__, tag=tag)
    tb_writer = create_tbwriter(os.path.join(log_dir, 'tb_log'), False)
    res = []
    global_avg_after_acc = 0.
    global_iter = 0
    
    for domain_index, _ in enumerate(scenarios[0].target_domains_order):
        avg_before_acc, avg_after_acc = 0., 0.
        cur_res = {}
        
        for task_name, online_model, scenario, da_alg, da_model, da_alg_hyp in zip(tasks_name, online_models, scenarios, da_algs, da_models, da_alg_hyps):

            cur_target_domain_name = scenario.target_domains_order[scenario.cur_domain_index]
            if cur_target_domain_name in da_alg_hyp:
                da_alg_hyp = da_alg_hyp[cur_target_domain_name]
                logger.info(f'use dataset-specific hyps')
            
            online_model.set_sd_sparsity(da_alg_hyp['sd_sparsity'])
            sd, unpruned_indexes_of_layers = online_model.generate_sd_by_target_samples(scenario.get_online_cur_domain_samples_for_training(da_alg_hyp['train_batch_size']))

            tmp_sd_path = os.path.join(log_dir, 'tmp_sd_model.pt')
            torch.save({'main': sd}, tmp_sd_path)
            
            if 'cls' not in task_name and 'pos' not in task_name and 'vqa' not in task_name:
                da_model_args = [f'{task_name}/{domain_index}', 
                        tmp_sd_path, 
                        device,
                        scenario.num_classes]
            else:
                da_model_args = [f'{task_name}/{domain_index}', 
                        tmp_sd_path, 
                        device]
            da_metrics, after_da_model = da_alg(
                {'main': da_model(*da_model_args)}, 
                os.path.join(log_dir, f'{task_name}/{domain_index}')
            ).run(scenario, {_k: _v for _k, _v in da_alg_hyp.items() if _k != 'sd_sparsity'})
            os.remove(tmp_sd_path)
            
            if domain_index > 0:
                shutil.rmtree(os.path.join(log_dir, f'{task_name}/{domain_index}/backup_codes'))

            online_model.sd_feedback_to_md(after_da_model['main'].models_dict['main'], unpruned_indexes_of_layers)
            online_model.md_feedback_to_self_fm()
            
            accs = da_metrics['accs']
            before_acc = accs[0]['acc']
            after_acc = accs[-1]['acc']
            
            avg_before_acc += before_acc
            avg_after_acc += after_acc
            
            for _acc in accs:
                tb_writer.add_scalar(f'total_acc', _acc['acc'], _acc['iter'] + global_iter) # TODO: bug here
            global_iter += _acc['iter'] + 1
            
            tb_writer.add_scalars(f'accs/{task_name}', dict(before=before_acc, after=after_acc), domain_index)
            tb_writer.add_scalar(f'times/{task_name}', da_metrics['time'], domain_index)
            
            scenario.next_domain()
            
            logger.info(f"task: {task_name}, domain {domain_index}, acc: {before_acc:.4f} -> "
                        f"{after_acc:.4f} ({da_metrics['time']:.2f}s)")
            cur_res[task_name] = da_metrics
        
        if involve_fm:
            for online_model in online_models:
                online_model.aggregate_fms_to_self_fm([m.models_dict['fm'] for m in online_models])
            for online_model in online_models:
                online_model.fm_feedback_to_md()
        
        avg_before_acc /= len(tasks_name)
        avg_after_acc /= len(tasks_name)
        tb_writer.add_scalars(f'accs/apps_avg', dict(before=avg_before_acc, after=avg_after_acc), domain_index)
        logger.info(f"--> domain {domain_index}, avg_acc: {avg_before_acc:.4f} -> "
                        f"{avg_after_acc:.4f}")
        res += [cur_res]
        
        global_avg_after_acc += avg_after_acc
        
        write_json(os.path.join(log_dir, 'res.json'), res, backup=False)

    global_avg_after_acc /= (domain_index + 1)
    logger.info(f'-----> final metric: {global_avg_after_acc:.4f}')
    write_json(os.path.join(log_dir, f'res_{global_avg_after_acc:.4f}.json'), res, backup=False)
    
    
    
def init_online_model(fm_models_dict_path, md_models_dict_path, task_name, __entry_file__):
    fm_models = torch.load(fm_models_dict_path)
    md_models = torch.load(md_models_dict_path)
    
    online_models_dict_path = save_models_dict_for_init({
        'fm': fm_models['main'],
        'md': md_models['main'],
        'sd': None,
        'indexes': md_models['indexes'],
        'bn_stats': md_models['bn_stats']
    }, __entry_file__, task_name)
    return online_models_dict_path
