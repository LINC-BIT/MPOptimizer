import sys
from utils.dl.common.env import set_random_seed
set_random_seed(1)

from typing import List
from data.dataloader import build_dataloader
from data import CLScenario, build_cl_scenario
from methods.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel

import torch
import sys
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from utils.dl.common.env import create_tbwriter
import os
from utils.common.log import logger
from utils.common.data_record import write_json
# from methods.shot.shot import OnlineShotModel
from methods.base.alg import BaseAlg
from methods.base.model import BaseModel


def baseline_cl(app_name: str,
                 scenario: CLScenario, 
                 cl_alg: BaseAlg, 
                 cl_alg_hyp: dict,
                 cl_model: BaseModel,
                 device,
                 __entry_file__,
                 tag=None):
    
    # involve_fm = settings['involve_fm']
    
    # task_name = app_name
    # online_model = elasticfm_model
    
    log_dir = get_res_save_dir(__entry_file__, tag=tag)
    tb_writer = create_tbwriter(os.path.join(log_dir, 'tb_log'), False)
    res = []
    global_avg_after_acc = 0.
    global_iter = 0
    
    for task_index, _ in enumerate(scenario.target_tasks_order):
        
        cur_target_task_name = scenario.target_tasks_order[scenario.cur_task_index]
        # if cur_target_domain_name in da_alg_hyp:
        #     da_alg_hyp = da_alg_hyp[cur_target_domain_name]
        #     logger.info(f'use dataset-specific hyps')
        
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
        
        cl_metrics, after_cl_model = cl_alg(
            {'main': cl_model}, 
            os.path.join(log_dir, f'{app_name}/{task_index}')
        ).run(scenario, cl_alg_hyp)
        # os.remove(tmp_sd_path)
        
        if task_index > 0:
            import shutil
            shutil.rmtree(os.path.join(log_dir, f'{app_name}/{task_index}/backup_codes'))
        
        accs = cl_metrics['accs']
        before_acc = accs[0]['acc']
        after_acc = accs[-1]['acc']
        
        tb_writer.add_scalars(f'accs/{app_name}', dict(before=before_acc, after=after_acc), task_index)
        tb_writer.add_scalar(f'times/{app_name}', cl_metrics['time'], task_index)
        
        for _acc in accs:
            tb_writer.add_scalar('total_acc', _acc['acc'], _acc['iter'] + global_iter)
        global_iter += _acc['iter'] + 1
        
        scenario.next_task()
        
        logger.info(f"app: {app_name}, task {task_index}, acc: {before_acc:.4f} -> "
                    f"{after_acc:.4f} ({cl_metrics['time']:.2f}s)")
        
        global_avg_after_acc += after_acc
        cur_res = cl_metrics
        res += [cur_res]
        write_json(os.path.join(log_dir, 'res.json'), res, backup=False)

    global_avg_after_acc /= (task_index + 1)
    logger.info(f'-----> final metric: {global_avg_after_acc:.4f}')
    write_json(os.path.join(log_dir, f'res_{global_avg_after_acc:.4f}.json'), res, backup=False)
    