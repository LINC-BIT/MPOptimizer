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
from copy import deepcopy
import time


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
            logger.info(f'use dataset-specific da_alg_hyp')
        
        da_metrics, after_da_model = da_alg(
            {'main': da_model}, 
            os.path.join(log_dir, f'{task_name}/{domain_index}')
        ).run(scenario, da_alg_hyp)
        # os.remove(tmp_sd_path)
        
        # 前面在当前域上训练，在这里压缩调优？
        # print(da_model.models_dict['main'])
        # 进行压缩
        reducing_width_ratio = 8
        samples = torch.rand(1, 3, 224, 224).to(device)

        trained_fm_model = deepcopy(da_model.models_dict['main'])
        fm_da_model = deepcopy(da_model) # 保存大模型
        lora_util = FMLoRA_ViT_Util()
        lora_absorbed_fm_model = lora_util.absorb_lora_and_recover_net_structure(trained_fm_model, samples)
        compressed_fm_model = FM_to_MD_ViT_Util().init_md_from_fm_by_reducing_width_with_perf_test(lora_absorbed_fm_model, reducing_width_ratio, samples)
        da_model.models_dict['main'] = compressed_fm_model
        
        # 进行调优？之前那个da_metrics是FM的结果吧，调优也能得到一个精度结果换成这个？
        
        datasets_for_training = scenario.get_online_cur_domain_datasets_for_training()
        train_dataset = datasets_for_training[cur_target_domain_name]['train']
        val_dataset = datasets_for_training[cur_target_domain_name]['val']
        datasets_for_inference = scenario.get_online_cur_domain_datasets_for_inference()
        test_dataset = datasets_for_inference
        
        train_loader = iter(build_dataloader(train_dataset, da_alg_hyp['train_batch_size'], da_alg_hyp['num_workers'], True, None))
        test_loader = build_dataloader(test_dataset, da_alg_hyp['val_batch_size'], da_alg_hyp['num_workers'], False, False)
        
        for p in compressed_fm_model.parameters():
            p.requires_grad = True
        da_model.to_train_mode()
        
        # 'distill_optimizer': 'AdamW',
        # 'distill_optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        optimizer = torch.optim.__dict__['AdamW']([
            {'params': da_model.models_dict['main'].parameters(), **{'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01}}
        ])
        if da_alg_hyp['scheduler'] != '':
            scheduler = torch.optim.lr_scheduler.__dict__[da_alg_hyp['scheduler']](optimizer, **da_alg_hyp['scheduler_args'])
        else:
            scheduler = None
        
        pbar = tqdm.tqdm(range(da_alg_hyp['num_iters']), dynamic_ncols=True)

        accs = []
        total_train_time = 0.
        cur_acc = 0.
        for iter_index in pbar:
            cur_start_time = time.time()
            da_model.to_train_mode()
            fm_da_model.to_eval_mode()
                
            x, y = next(train_loader)
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
                y = y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                fm_output = fm_da_model.infer(x)
            md_output = da_model.infer(x)
            
            distill_criterion = CrossEntropyLossSoft()
            total_loss = distill_criterion(md_output, fm_output)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_train_time += time.time() - cur_start_time
                
            if (iter_index + 1) % da_alg_hyp['val_freq'] == 0:
                from data import split_dataset
                cur_md = da_model.models_dict['main']
                md_for_test = deepcopy(da_model.models_dict['main'])
                da_model.models_dict['main'] = md_for_test
                cur_test_batch_dataset = split_dataset(test_dataset, da_alg_hyp['val_batch_size'], iter_index + 1)[0]
                cur_test_batch_dataloader = build_dataloader(cur_test_batch_dataset, da_alg_hyp['train_batch_size'], da_alg_hyp['num_workers'], False, False)
                da_model.to_eval_mode()
                cur_acc = da_model.get_accuracy(cur_test_batch_dataloader)
                accs += [{
                    'iter': iter_index + 1,
                    'acc': cur_acc
                }]
            pbar.set_description(f'loss: {total_loss:.6f}, cur_acc: {cur_acc:.4f}')
            
        time_usage = total_train_time
        da_metrics = {
            'accs': accs,
            'time': time_usage
        }     
        da_model = fm_da_model # 恢复大模型
                
        # 蒸馏结束
        
        
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
    