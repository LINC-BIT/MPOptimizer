# from typing import Any, Dict
# from schema import Schema, Or
# import schema
# from data import Scenario, MergedDataset
# from methods.base.alg import BaseAlg
# from data import build_dataloader
# from ..model import ElasticDNN_OfflineFMModel, ElasticDNN_OfflineMDModel
# from ...model.base import ElasticDNNUtil
# import torch.optim
# import tqdm
# import torch.nn.functional as F
# from torch import nn
# from utils.dl.common.env import create_tbwriter
# import os
# import random
# import numpy as np
# from copy import deepcopy
# from utils.dl.common.model import LayerActivation2, get_module
# from utils.common.log import logger


# class ElasticDNN_MDPretrainingWoFBSAlg(BaseAlg):
#     """
#     TODO: fine-tuned FM -> init MD -> trained MD -> construct indexes (only between similar weights) and fine-tune
#     """
#     def get_required_models_schema(self) -> Schema:
#         return Schema({
#             'fm': ElasticDNN_OfflineFMModel,
#             'md': ElasticDNN_OfflineMDModel
#         })
        
#     def get_required_hyp_schema(self) -> Schema:
#         return Schema({
#             'launch_tbboard': bool,
            
#             'samples_size': object,
#             'generate_md_width_ratio': int,
            
#             'train_batch_size': int,
#             'val_batch_size': int,
#             'num_workers': int,            
#             'optimizer': str,
#             'optimizer_args': dict,
#             'scheduler': str,
#             'scheduler_args': dict,
#             'num_iters': int,
#             'val_freq': int,
#             'distill_loss_weight': float
#         })

#     def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
#         super().run(scenario, hyps)
        
#         assert isinstance(self.models['md'], ElasticDNN_OfflineMDModel) # for auto completion
#         assert isinstance(self.models['fm'], ElasticDNN_OfflineFMModel) # for auto completion
        
#         # 1. add FBS
#         device = self.models['md'].device
        
#         if self.models['md'].models_dict['main'] == -1:
#             logger.info(f'init master DNN by reducing width of an adapted foundation model (already tuned by LoRA)...')
            
#             before_fm_model = deepcopy(self.models['fm'].models_dict['main'])
#             lora_util = self.models['fm'].get_lora_util()
            
#             sample = hyps['samples_size']
#             if isinstance(sample, (tuple, list)) and isinstance(sample[0], int):
#                 sample = torch.rand(hyps['samples_size']).to(device)
                
#             lora_absorbed_fm_model = lora_util.absorb_lora_and_recover_net_structure(self.models['fm'].models_dict['main'], 
#                                                                                      sample)
#             self.models['fm'].models_dict['main'] = lora_absorbed_fm_model
#             master_dnn = self.models['fm'].generate_md_by_reducing_width(hyps['generate_md_width_ratio'], 
#                                                                          sample)
#             self.models['fm'].models_dict['main'] = before_fm_model
            
#             self.models['md'].models_dict['main'] = master_dnn
#             self.models['md'].to(device)
            
        
        
#         # 2. train (knowledge distillation, index relationship)
#         offline_datasets = scenario.get_offline_datasets()
#         train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
#         val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
#         train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
#                                         True, None))
#         val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
#                                       False, False)
        
#         # val_acc = self.models['md'].get_accuracy(val_loader)
#         # print(val_acc)
#         # exit()
        
#         # 2.1 train whole master DNN (knowledge distillation)
#         self.models['md'].to_train_mode()
#         for p in master_dnn.parameters():
#             p.requires_grad = True
        
#         if hasattr(self.models['md'], 'get_trained_params'):
#             trained_p = self.models['md'].get_trained_params()
#             logger.info(f'use custom trained parameters!!')
#         else:
#             trained_p = self.models['md'].models_dict['main'].parameters()
#         for p in trained_p:
#             p.requires_grad = True
#         optimizer = torch.optim.__dict__[hyps['optimizer']]([
#             {'params': trained_p, **hyps['optimizer_args']}
#         ])
#         scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
#         tb_writer = create_tbwriter(os.path.join(self.res_save_dir, 'tb_log'), launch_tbboard=hyps['launch_tbboard'])
#         pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True)
#         best_avg_val_acc = 0.
        
#         md_output_hook = None
        
#         for iter_index in pbar:
#             self.models['md'].to_train_mode()
#             self.models['fm'].to_eval_mode()
            
#             # rand_sparsity = random.random() * (hyps['max_sparsity'] - hyps['min_sparsity']) + hyps['min_sparsity']
#             # elastic_dnn_util.set_master_dnn_sparsity(self.models['md'].models_dict['main'], rand_sparsity)
#             if md_output_hook is None:
#                 md_output_hook = self.models['md'].get_feature_hook()
#                 fm_output_hook = self.models['fm'].get_feature_hook()
                
#             x, y = next(train_loader)
#             if isinstance(x, dict):
#                 for k, v in x.items():
#                     if isinstance(v, torch.Tensor):
#                         x[k] = v.to(device)
#                 y = y.to(device)
#             else:
#                 x, y = x.to(device), y.to(device)
            
#             with torch.no_grad():
#                 fm_output = self.models['fm'].infer(x)
#             task_loss = self.models['md'].forward_to_get_task_loss(x, y)
            
#             if isinstance(md_output_hook, (tuple, list)):
#                 distill_loss = 0.
#                 for h1, h2 in zip(md_output_hook, fm_output_hook):
#                     md_output = h1.output
#                     fm_output = h2.output
#                     distill_loss += hyps['distill_loss_weight'] * self.models['md'].get_distill_loss(md_output, fm_output)
#             else:
#                 md_output = md_output_hook.output
#                 fm_output = fm_output_hook.output
#                 distill_loss = hyps['distill_loss_weight'] * self.models['md'].get_distill_loss(md_output, fm_output)

#             total_loss = task_loss + distill_loss
            
#             optimizer.zero_grad()
#             total_loss.backward()
            
#             # for n, p in self.models['md'].models_dict['main'].named_parameters():
#             #     if p.grad is not None:
#             #         print(n)
#             # exit()
            
#             optimizer.step()
#             scheduler.step()
            
#             if (iter_index + 1) % hyps['val_freq'] == 0:
                
#                 # elastic_dnn_util.clear_cached_channel_attention_in_master_dnn(self.models['md'].models_dict['main'])
#                 if isinstance(md_output_hook, (tuple, list)):
#                     [h.remove() for h in md_output_hook]
#                     [h.remove() for h in fm_output_hook]
#                 else:
#                     md_output_hook.remove()
#                     fm_output_hook.remove()
                
#                 md_output_hook = None
#                 fm_output_hook = None
                
#                 cur_md = self.models['md'].models_dict['main']
#                 md_for_test = deepcopy(self.models['md'].models_dict['main'])
#                 val_acc = 0.
                
#                 self.models['md'].models_dict['main'] = md_for_test
#                 self.models['md'].to_eval_mode()
#                 val_acc = self.models['md'].get_accuracy(val_loader)
                
#                 self.models['md'].models_dict['main'] = cur_md
                
#                 self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_last.pt'))
#                 self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_last.pt'))
                
#                 if val_acc > best_avg_val_acc:
#                     best_avg_val_acc = val_acc
#                     self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_best.pt'))
#                     self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_best.pt'))
                
#             tb_writer.add_scalars(f'losses', dict(task=task_loss, distill=distill_loss, total=total_loss), iter_index)
#             pbar.set_description(f'loss: {total_loss:.6f}')
#             if (iter_index + 1) >= hyps['val_freq']:
#                 tb_writer.add_scalar(f'accs/val_acc', val_acc, iter_index)
#                 pbar.set_description(f'loss: {total_loss:.6f}, val_acc: {val_acc:.4f}')
            
# code below is commented on 0716 17:49, because of a bug that the loss cannot be gradient decented 
# (bug confirmed, why? I dont know :)
            
            
            
            
            
            
            
            
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
from utils.dl.common.model import LayerActivation2, get_module
from utils.common.log import logger
from data import split_dataset


class ElasticDNN_MDPretrainingWoFBSAlg(BaseAlg):
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
            
            'samples_size': any,
            'generate_md_width_ratio': int,
            
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int,            
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'num_iters': int,
            'val_freq': int,
            'distill_loss_weight': float
        })

    def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
        super().run(scenario, hyps)
        
        assert isinstance(self.models['md'], ElasticDNN_OfflineMDModel) # for auto completion
        assert isinstance(self.models['fm'], ElasticDNN_OfflineFMModel) # for auto completion
        
        # 1. add FBS
        device = self.models['md'].device
        
        if self.models['md'].models_dict['main'] == -1:
            logger.info(f'init master DNN by reducing width of an adapted foundation model (already tuned by LoRA)...')
            
            before_fm_model = deepcopy(self.models['fm'].models_dict['main'])
            lora_util = self.models['fm'].get_lora_util()
            
            sample = hyps['samples_size']
            if isinstance(sample, (tuple, list)) and isinstance(sample[0], int):
                sample = torch.rand(hyps['samples_size']).to(device)
            
            lora_absorbed_fm_model = lora_util.absorb_lora_and_recover_net_structure(self.models['fm'].models_dict['main'], 
                                                                                    sample)
            self.models['fm'].models_dict['main'] = lora_absorbed_fm_model
            master_dnn = self.models['fm'].generate_md_by_reducing_width(hyps['generate_md_width_ratio'], 
                                                                        sample)
            self.models['fm'].models_dict['main'] = before_fm_model
            
            self.models['md'].models_dict['main'] = master_dnn
            self.models['md'].to(device)
        
        # 2. train (knowledge distillation, index relationship)
        offline_datasets = scenario.get_offline_datasets()
        train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
        val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
        train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                        True, None))
        val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
                                      False, False)
        
        # logger.info(f'FM acc: {self.models["fm"].get_accuracy(val_loader):.4f}')
        
        # 2.1 train whole master DNN (knowledge distillation)
        
        # self.models['md'].models_dict['main'] = master_dnn = nn.DataParallel(master_dnn.to(device), device_ids=['cuda:0'], output_device=['cuda:0'])
        
        for p in master_dnn.model.parameters():
            p.requires_grad = True
        for p in master_dnn.model.text_model.parameters():
            p.requires_grad = False
        self.models['md'].to_train_mode()
        
        optimizer = torch.optim.__dict__[hyps['optimizer']]([
            {'params': [p for n, p in self.models['md'].models_dict['main'].named_parameters() if 'text_model' not in n], **hyps['optimizer_args']}
        ])
        scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
        tb_writer = create_tbwriter(os.path.join(self.res_save_dir, 'tb_log'), launch_tbboard=hyps['launch_tbboard'])
        pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True)
        best_avg_val_acc = 0.
        
        md_output_hook = None
        
        train_ratio = 0.005
        
        for iter_index in pbar:
            
            if iter_index % 500 == 0:
                train_ratio += 0.005
                
                train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
                train_dataset = split_dataset(train_dataset, max(hyps['train_batch_size'], int(train_ratio * len(train_dataset))))[0]
                train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'], True, None))
            
            self.models['md'].to_train_mode()
            # self.models['fm'].to_eval_mode()
            
            # rand_sparsity = random.random() * (hyps['max_sparsity'] - hyps['min_sparsity']) + hyps['min_sparsity']
            # elastic_dnn_util.set_master_dnn_sparsity(self.models['md'].models_dict['main'], rand_sparsity)
            if md_output_hook is None:
                md_output_hook = self.models['md'].get_feature_hook()
                fm_output_hook = self.models['fm'].get_feature_hook()
                
            x, y = next(train_loader)
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
                y = y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            
            # with torch.no_grad():
            #     fm_output = self.models['fm'].infer(x)
            # task_loss = self.models['md'].forward_to_get_task_loss(x, y)
            task_loss = 0.
            with torch.no_grad():
                fm_output = self.models['fm'].forward_to_get_task_loss(x, y)
            md_output = self.models['md'].forward_to_get_task_loss(x, y)
            
            if iter_index % 10 == 0:
                print(fm_output, md_output)
            # md_output = md_output_hook.output
            # fm_output = fm_output_hook.output
            
            distill_loss = hyps['distill_loss_weight'] * self.models['md'].get_distill_loss(md_output, fm_output)
            # distill_loss = 0.
            
            # logit_diff_loss = -((md_output - md_output.mean(dim=1).unsqueeze(1)) ** 2).sum() * 0.01
            logit_diff_loss = 0.
            
            total_loss = task_loss + distill_loss + logit_diff_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (iter_index + 1) % hyps['val_freq'] == 0:
                
                # elastic_dnn_util.clear_cached_channel_attention_in_master_dnn(self.models['md'].models_dict['main'])
                md_output_hook.remove()
                md_output_hook = None
                fm_output_hook.remove()
                fm_output_hook = None
                
                cur_md = self.models['md'].models_dict['main']
                md_for_test = deepcopy(self.models['md'].models_dict['main'])
                val_acc = 0.
                
                self.models['md'].models_dict['main'] = md_for_test
                self.models['md'].to_eval_mode()
                val_acc = self.models['md'].get_accuracy(val_loader)
                
                self.models['md'].models_dict['main'] = cur_md
                
                self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_last.pt'))
                self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_last.pt'))
                
                if val_acc > best_avg_val_acc:
                    best_avg_val_acc = val_acc
                    self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_best.pt'))
                    self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_best.pt'))
                
            tb_writer.add_scalars(f'losses', dict(task=task_loss, distill=distill_loss, logit_diff_loss=logit_diff_loss, total=total_loss), iter_index)
            # tb_writer.add_scalar(f'lr', optimizer.param_groups[0]['lr'], iter_index)
            tb_writer.add_scalar(f'lr', optimizer.param_groups[0]['lr'], iter_index)
            tb_writer.add_scalar(f'train_data_size', len(train_dataset), iter_index)
            
            pbar.set_description(f'loss: {total_loss:.6f}, logit_diff_loss: {logit_diff_loss:.6f}')
            if (iter_index + 1) >= hyps['val_freq']:
                tb_writer.add_scalar(f'accs/val_acc', val_acc, iter_index)
                pbar.set_description(f'loss: {total_loss:.6f}, logit_diff_loss: {logit_diff_loss:.6f}, val_acc: {val_acc:.4f}')
            