from typing import Any, Dict
from schema import Schema, Or
import schema
from data import Scenario, MergedDataset
from methods.base.alg import BaseAlg
from data import build_dataloader
from ..model import ElasticDNN_OfflineFMModel, ElasticDNN_OfflineMDModel
from ...model.base import ElasticDNNUtil
import torch.optim
import tqdm
import torch.nn.functional as F
from torch import nn
from utils.dl.common.env import create_tbwriter
import os
import random
import numpy as np
from copy import deepcopy
from utils.dl.common.model import LayerActivation, get_module
from utils.common.log import logger


class ElasticDNN_MDPretrainingWFBSAlg(BaseAlg):
    """
    TODO: fine-tuned FM -> init MD -> trained MD -> construct indexes (only between similar weights) and fine-tune
    """
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'fm': ElasticDNN_OfflineFMModel,
            'md': ElasticDNN_OfflineMDModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        return Schema({
            'launch_tbboard': bool,
            
            'samples_size': (int, int, int, int),
            'generate_md_width_ratio': int,
            
            'FBS_r': int,
            'FBS_ignore_layers': [str],
            
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int,            
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'num_iters': int,
            'val_freq': int,
            'max_sparsity': float,
            'min_sparsity': float,
            'l1_reg_loss_weight': float,
            'val_num_sparsities': int,
            
            'bn_cal_num_iters': int
        })
        
    def bn_cal(self, model: nn.Module, train_loader, num_iters, device):
        has_bn = False
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                has_bn = True
                break
        
        if not has_bn:
            return {}
        
        def bn_calibration_init(m):
            """ calculating post-statistics of batch normalization """
            if getattr(m, 'track_running_stats', False):
                # reset all values for post-statistics
                m.reset_running_stats()
                # set bn in training mode to update post-statistics
                m.training = True
                
        with torch.no_grad():
            model.eval()
            model.apply(bn_calibration_init)
            for _ in range(num_iters):
                x, _ = next(train_loader)
                model(x.to(device))
            model.eval()
            
        bn_stats = {}
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_stats[n] = m
        return bn_stats
        
    def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
        super().run(scenario, hyps)
        
        assert isinstance(self.models['md'], ElasticDNN_OfflineMDModel) # for auto completion
        assert isinstance(self.models['fm'], ElasticDNN_OfflineFMModel) # for auto completion
        
        # 1. add FBS
        device = self.models['md'].device
        
        # 2. train (knowledge distillation, index relationship)
        offline_datasets = scenario.get_offline_datasets()
        train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
        val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
        train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                        True, None))
        val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
                                      False, False)
        
        logger.info(f'master DNN acc before inserting FBS: {self.models["md"].get_accuracy(val_loader):.4f}')
        
        master_dnn = self.models['md'].models_dict['main']
        elastic_dnn_util = self.models['fm'].get_elastic_dnn_util()
        master_dnn = elastic_dnn_util.convert_raw_dnn_to_master_dnn_with_perf_test(master_dnn, hyps['FBS_r'], hyps['FBS_ignore_layers']).to(device)
        self.models['md'].models_dict['main'] = master_dnn
        
        # 2.1 train whole master DNN (knowledge distillation)
        for p in master_dnn.parameters():
            p.requires_grad = True
        self.models['md'].to_train_mode()
        
        optimizer = torch.optim.__dict__[hyps['optimizer']]([
            {'params': self.models['md'].models_dict['main'].parameters(), **hyps['optimizer_args']}
        ])
        scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
        tb_writer = create_tbwriter(os.path.join(self.res_save_dir, 'tb_log'), launch_tbboard=hyps['launch_tbboard'])
        pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True)
        best_avg_val_acc = 0.
        
        for iter_index in pbar:
            self.models['md'].to_train_mode()
            self.models['fm'].to_eval_mode()
            
            rand_sparsity = random.random() * (hyps['max_sparsity'] - hyps['min_sparsity']) + hyps['min_sparsity']
            elastic_dnn_util.set_master_dnn_sparsity(self.models['md'].models_dict['main'], rand_sparsity)
            
            x, y = next(train_loader)
            x, y = x.to(device), y.to(device)
            
            task_loss = self.models['md'].forward_to_get_task_loss(x, y)
            l1_reg_loss = hyps['l1_reg_loss_weight'] * elastic_dnn_util.get_accu_l1_reg_of_raw_channel_attention_in_master_dnn(master_dnn)
            total_loss = task_loss + l1_reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (iter_index + 1) % hyps['val_freq'] == 0:
                
                elastic_dnn_util.clear_cached_channel_attention_in_master_dnn(self.models['md'].models_dict['main'])
                
                cur_md = self.models['md'].models_dict['main']
                md_for_test = deepcopy(self.models['md'].models_dict['main'])
                val_accs = {}
                avg_val_acc = 0.
                bn_stats = {}
                
                for val_sparsity in np.linspace(hyps['min_sparsity'], hyps['max_sparsity'], num=hyps['val_num_sparsities']):
                    elastic_dnn_util.set_master_dnn_sparsity(md_for_test, val_sparsity)
                    bn_stats[f'{val_sparsity:.4f}'] = self.bn_cal(md_for_test, train_loader, hyps['bn_cal_num_iters'], device)
                    
                    # generate seperate surrogate DNN
                    test_sd = elastic_dnn_util.extract_surrogate_dnn_via_samples_with_perf_test(md_for_test, x)
                    
                    self.models['md'].models_dict['main'] = test_sd
                    self.models['md'].to_eval_mode()
                    val_acc = self.models['md'].get_accuracy(val_loader)
                    
                    val_accs[f'{val_sparsity:.4f}'] = val_acc
                    avg_val_acc += val_acc
                    
                avg_val_acc /= hyps['val_num_sparsities']
                
                self.models['md'].models_dict['main'] = cur_md
                self.models['md'].models_dict['bn_stats'] = bn_stats
                
                self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_last.pt'))
                self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_last.pt'))
                
                if avg_val_acc > best_avg_val_acc:
                    best_avg_val_acc = avg_val_acc
                    self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_best.pt'))
                    self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_best.pt'))
                
            tb_writer.add_scalars(f'losses', dict(task=task_loss, l1_reg=l1_reg_loss, total=total_loss), iter_index)
            pbar.set_description(f'loss: {total_loss:.6f}')
            if (iter_index + 1) >= hyps['val_freq']:
                tb_writer.add_scalars(f'accs/val_accs', val_accs, iter_index)
                tb_writer.add_scalar(f'accs/avg_val_acc', avg_val_acc, iter_index)
                pbar.set_description(f'loss: {total_loss:.6f}, task_loss: {task_loss:.6f}, '
                                     f'l1_loss: {l1_reg_loss:.6f}, avg_val_acc: {avg_val_acc:.4f}')
            