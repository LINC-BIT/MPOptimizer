from typing import Any, Dict
from schema import Schema
from data import Scenario, MergedDataset
from new_impl.cv.base.alg import BaseAlg
from data import build_dataloader
from ..model import ElasticDNN_OfflineFMModel
from ...model.base import ElasticDNNUtil
import torch.optim
import tqdm
from torch import nn
from utils.dl.common.env import create_tbwriter
import os
import random
import numpy as np
from copy import deepcopy
from utils.dl.common.model import get_module
from utils.common.log import logger


class ElasticDNN_FMLoRAAlg(BaseAlg):
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'fm': ElasticDNN_OfflineFMModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        from schema import Optional
        return Schema({
            'launch_tbboard': bool,
            
            'samples_size': object,
            'ab_r': int,
            
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'num_iters': int,
            'val_freq': int,
            
            Optional('fm_lora_ckpt_path'): str
        })
        
    def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
        super().run(scenario, hyps)
        assert isinstance(self.models['fm'], ElasticDNN_OfflineFMModel) # for auto completion

        # 1. add LoRA
        lora_util = self.models['fm'].get_lora_util()
        device = self.models['fm'].device

        sample = hyps['samples_size']
        if isinstance(sample, (tuple, list)) and isinstance(sample[0], int):
            sample = torch.rand(hyps['samples_size']).to(device)
        lora_util.add_lora_ab_to_fm(self.models['fm'].models_dict['main'], hyps['ab_r'], sample)
        
        if 'fm_lora_ckpt_path' in hyps.keys() and hyps['fm_lora_ckpt_path'] != '' and hyps['fm_lora_ckpt_path'] is not None:
            _ckpt = torch.load(hyps['fm_lora_ckpt_path'])['main']

            new_state_dict = deepcopy(self.models['fm'].models_dict['main'].state_dict())
            
            for n, p in _ckpt.named_parameters():
                if 'qkv.abs' not in n:
                    continue
                
                new_state_dict[n] = p
                logger.info(f'use {n} from ckpt')
            
            self.models['fm'].models_dict['main'].load_state_dict(new_state_dict)
           

        # 2. train (knowledge distillation, index relationship)
        offline_datasets = scenario.get_offline_datasets()
        train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
        
        # debug
        # from data.visualize import visualize_classes_in_object_detection
        # d = offline_datasets['GTA5Det']['val']
        # class_to_idx_map = {c: d.idx_map[i] for i, c in enumerate(d.classes)}
        # print(class_to_idx_map)
        # visualize_classes_in_object_detection(d, class_to_idx_map,
        #                                       {}, os.path.join(self.res_save_dir, 'debug.png'))
        # exit()
        val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
        train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                        True, None))
        # train_loader = build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
        #                                 True, None)
        # if hyps['use_train_loader_for_val']:
        #     val_loader = build_dataloader(train_dataset, hyps['val_batch_size'], hyps['num_workers'],
        #                               False, False)
        #     logger.warn('use train loader for val!!!')
        # else:
        val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
                                    False, False)
        
        lora_params = lora_util.train_only_lora(self.models['fm'].models_dict['main'])
        # x = torch.rand(1,3,224,224).to('cuda')            
        # print(self.models['fm'].models_dict['main'](x).logits) 
        head_params = self.models['fm'].get_task_head_params()
        optimizer = torch.optim.__dict__[hyps['optimizer']](lora_params + head_params, **hyps['optimizer_args'])
        scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
        
        fbs_tb_writer = create_tbwriter(os.path.join(self.res_save_dir, 'tb_log'), launch_tbboard=hyps['launch_tbboard'])
        pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True)
        
        best_val_acc = 0
        val_acc = 0
        
        for iter_index in pbar:
            self.models['fm'].to_train_mode()
            #x, y, _ = next(train_loader)
            x, y  = next(train_loader)
            # x, y = train_loader
            if isinstance(x, dict) and isinstance(y,torch.Tensor):
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
                y = y.to(device)
            elif isinstance(x, dict) and isinstance(y,dict):
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
                for k,v in y.items():
                    if isinstance(v,torch.Tensor):
                        y[k] = v.to(device)
            else:
                x, y = x.to(device), y.to(device)
            task_loss = self.models['fm'].forward_to_get_task_loss(x, y)
            #task_loss.requires_grad_(True)
            optimizer.zero_grad()
            task_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (iter_index + 1) % hyps['val_freq'] == 0:
                # logger.warn('use train loader for val!!!')
                
                self.models['fm'].to_eval_mode()
                val_acc = self.models['fm'].get_accuracy(val_loader)
                
                self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_last.pt'))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_best.pt'))
                
            fbs_tb_writer.add_scalar(f'losses/task_loss', task_loss, iter_index)
            fbs_tb_writer.add_scalar(f'accs/val_acc', val_acc, iter_index)
            fbs_tb_writer.add_scalar(f'lr', optimizer.param_groups[0]['lr'], iter_index)
            pbar.set_description(f'loss: {task_loss:.6f}, val_acc: {val_acc:.4f}')
