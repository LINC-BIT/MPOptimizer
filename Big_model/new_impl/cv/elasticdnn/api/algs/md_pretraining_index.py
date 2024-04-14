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
import matplotlib.pyplot as plt


class ElasticDNN_MDPretrainingIndexAlg(BaseAlg):
    """
    construct indexes between a filter/row of MD and all filters/rows of FM in the same layer
    too huge indexes (~1GB), train so slow, hard to optimize
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
            
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int,            
            'optimizer': str,
            'indexes_optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'num_iters': int,
            'val_freq': int,
            'index_loss_l1_weight': float,
            'val_num_sparsities': int,
            
            'bn_cal_num_iters': int,
            
            'index_guided_linear_comb_split_size': Or(int, None)
        })
        
    def upsample_2d_tensor(self, p: torch.Tensor, target_len: int):
        assert p.dim() == 2 # regard 2d weight as (batch_size, 1d_vector_dim)
        return F.upsample(p.unsqueeze(1).unsqueeze(3),
                          size=(target_len, 1),
                          mode='bilinear').squeeze(3).squeeze(1)
        
    def two_params_diff_fast(self, trained_p: torch.Tensor, ref_p: torch.Tensor, 
                             index: torch.Tensor, 
                             split_size: int):

        assert trained_p.dim() == ref_p.dim()
        assert index.size(0) == trained_p.size(0) and index.size(1) == ref_p.size(0)
        
        # print(trained_p.size(), ref_p.size(), index.size())

        ref_p = ref_p.detach()
        if trained_p.dim() > 1:
            trained_p = trained_p.flatten(1)
            ref_p = ref_p.flatten(1)
            
            # the weight size of master DNN and foundation model may be totally different
            
            # MD -> FM: upsample first
            # FM -> MD: downsample first
            if trained_p.size(1) < ref_p.size(1):
                trained_p = self.upsample_2d_tensor(trained_p, ref_p.size(1))
            
            index = index.unsqueeze(-1)
        #     linear_combed_ref_p = (ref_p.unsqueeze(0) * index).sum(1)
        # else:
        
        # print(trained_p.size(), ref_p.size(), index.size())
        
        if split_size is None:
            # old version: huge memory consumption, not recommended (although this is fastest)
            # print('old version')
            linear_combed_ref_p = (ref_p.unsqueeze(0) * index).sum(1)
        
        else:
            # new version
            linear_combed_ref_p = 0
            
            cur_split_size = split_size
            while index.size(1) % cur_split_size != 0:
                cur_split_size -= 1
            # print(cur_split_size) 
            
            for i in range(0, index.size(1), cur_split_size):
                # if not isinstance(linear_combed_ref_p, int):
                    # print(linear_combed_ref_p.size(), ref_p.unsqueeze(0)[:, i: i + cur_split_size].size(), index[:, i: i + cur_split_size].size())
                linear_combed_ref_p += ref_p.unsqueeze(0)[:, i: i + cur_split_size] * index[:, i: i + cur_split_size]
            linear_combed_ref_p = linear_combed_ref_p.sum(1)
            
        diff = (linear_combed_ref_p - trained_p).norm(2) ** 2
        return diff
        
    def get_index_loss(self, fm, md, indexes, match_fn, split_size):
        res = 0.

        for name, p in md.named_parameters():
            if name not in indexes.keys():
                continue
            # if p.dim() == 0:
            #     continue
            
            raw_p = match_fn(name, fm)
            # if raw_p is None:
            #     continue

            index = indexes[name]
            
            # print(name)
            res += self.two_params_diff_fast(p, raw_p, index, split_size)
        return res
    
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
        
        # sanity check
        # a=  torch.tensor([[1, 2, 3], [1, 2, 4]])
        # index = torch.tensor([[1, 2, 3],
        # [1, 2, 4]])
        # b = torch.tensor([[1, 2, 3], [1, 2, 4], [2, 3, 4]])
        # print(self.two_params_diff_fast(a, b, index, hyps['index_guided_linear_comb_split_size']))
        
        assert isinstance(self.models['md'], ElasticDNN_OfflineMDModel) # for auto completion
        assert isinstance(self.models['fm'], ElasticDNN_OfflineFMModel) # for auto completion
        
        # 1. add FBS
        device = self.models['md'].device
        
        # logger.info(f'init master DNN by reducing width of an adapted foundation model (already tuned by LoRA)...')
        
        # before_fm_model = deepcopy(self.models['fm'].models_dict['main'])
        # lora_util = self.models['fm'].get_lora_util()
        # lora_absorbed_fm_model = lora_util.absorb_lora_and_recover_net_structure(self.models['fm'].models_dict['main'], 
        #                                                                          torch.rand(hyps['samples_size']).to(device))
        # self.models['fm'].models_dict['main'] = lora_absorbed_fm_model
        # master_dnn = self.models['fm'].generate_md_by_reducing_width(hyps['generate_md_width_ratio'], 
        #                                                              torch.rand(hyps['samples_size']).to(device))
        # self.models['fm'].models_dict['main'] = before_fm_model
        
        elastic_dnn_util = self.models['fm'].get_elastic_dnn_util()
        # master_dnn = elastic_dnn_util.convert_raw_dnn_to_master_dnn_with_perf_test(master_dnn,
        #                                                                            hyps['FBS_r'], hyps['FBS_ignore_layers'])
        # self.models['md'].models_dict['main'] = master_dnn
        # self.models['md'].to(device)
        # master_dnn = self.models['md'].models_dict['main']
        
        # 2. train (knowledge distillation, index relationship)
        offline_datasets = scenario.get_offline_datasets()
        train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
        val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
        train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                                        True, None))
        val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
                                      False, False)
        
        # 2.1 train only FBS (skipped because current md cannot do proper inference)
            
        # 2.2 train whole master DNN (knowledge distillation, index relationship)
        # for p in master_dnn.parameters():
        #     p.requires_grad = True
        # self.models['md'].to_train_mode()
        
        indexes = {}
        for name, p in self.models['md'].models_dict['main'].named_parameters():
            if p.dim() > 1:
                matched_p_in_fm = self.models['md'].get_matched_param_of_fm(name, self.models['fm'].models_dict['main'])
                if matched_p_in_fm is None:
                    continue
                indexes[name] = torch.zeros((p.size(0), matched_p_in_fm.size(0))).to(device)
                indexes[name].requires_grad = True
                
                logger.info(f'construct index in layer {name}')
        
        tmp_indexes_file_path = os.path.join(self.res_save_dir, 'tmp-indexes.pt')
        torch.save(indexes, tmp_indexes_file_path)
        logger.info(f'generate indexes ({(os.path.getsize(tmp_indexes_file_path) / 1024**2):.3f}MB)')
        os.remove(tmp_indexes_file_path)
        
        optimizer = torch.optim.__dict__[hyps['optimizer']]([
            # {'params': self.models['md'].models_dict['main'].parameters(), **hyps['optimizer_args']},
            {'params': [v for v in indexes.values()], **hyps['indexes_optimizer_args']}
        ])
        scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
        tb_writer = create_tbwriter(os.path.join(self.res_save_dir, 'tb_log'), launch_tbboard=hyps['launch_tbboard'])
        pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True)
        best_avg_val_acc = 0.
        
        for p in self.models['md'].models_dict['main'].parameters():
            p.requires_grad = False
        for p in self.models['fm'].models_dict['main'].parameters():
            p.requires_grad = False
        
        for iter_index in pbar:
            self.models['md'].to_eval_mode()
            self.models['fm'].to_eval_mode()
            
            # rand_sparsity = random.random() * (hyps['max_sparsity'] - hyps['min_sparsity']) + hyps['min_sparsity']
            # elastic_dnn_util.set_master_dnn_sparsity(self.models['md'].models_dict['main'], rand_sparsity)
            
            # x, y = next(train_loader)
            # x, y = x.to(device), y.to(device)
            
            # task_loss = self.models['md'].forward_to_get_task_loss(x, y)
            # l1_reg_loss = hyps['l1_reg_loss_weight'] * elastic_dnn_util.get_accu_l1_reg_of_raw_channel_attention_in_master_dnn(master_dnn)
            index_loss = self.get_index_loss(self.models['fm'].models_dict['main'], 
                                                                         self.models['md'].models_dict['main'], 
                                                                         indexes,
                                                                         self.models['md'].get_matched_param_of_fm,
                                                                         hyps['index_guided_linear_comb_split_size'])
            index_l1_loss = hyps['index_loss_l1_weight'] * torch.FloatTensor([v.abs().sum() for v in indexes.values()]).sum()
            
            total_loss = index_loss + index_l1_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (iter_index + 1) % (10) == 0:
                # visualize indexes histgoram
                # do not use add_histogram because indexes is huge
                os.makedirs(os.path.join(self.res_save_dir, 'index_hist'), exist_ok=True)
                with torch.no_grad():
                    for p_name, index in indexes.items():
                        index_hist = index.view(-1).histc(bins=20).detach().cpu().numpy()
                        plt.bar(list(range(20)), index_hist)
                        plt.savefig(os.path.join(self.res_save_dir, f'index_hist/{p_name}.png'))
                        plt.clf()
            
            if (iter_index + 1) % hyps['val_freq'] == 0:
                
                elastic_dnn_util.clear_cached_channel_attention_in_master_dnn(self.models['md'].models_dict['main'])
                
                cur_md = self.models['md'].models_dict['main']
                md_for_test = deepcopy(self.models['md'].models_dict['main'])
                val_accs = {}
                avg_val_acc = 0.
                bn_stats = {}
                
                for val_sparsity in [0.0, 0.2, 0.4, 0.8]:
                    elastic_dnn_util.set_master_dnn_sparsity(md_for_test, val_sparsity)
                    bn_stats[f'{val_sparsity:.4f}'] = self.bn_cal(md_for_test, train_loader, hyps['bn_cal_num_iters'], device)
                    self.models['md'].models_dict['main'] = md_for_test
                    self.models['md'].to_eval_mode()
                    val_acc = self.models['md'].get_accuracy(val_loader)
                    
                    val_accs[f'{val_sparsity:.4f}'] = val_acc
                    avg_val_acc += val_acc
                avg_val_acc /= hyps['val_num_sparsities']
                    
                self.models['md'].models_dict['main'] = cur_md
                self.models['md'].models_dict['indexes'] = indexes
                self.models['md'].models_dict['bn_stats'] = bn_stats
                self.models['fm'].models_dict['indexes'] = indexes
                
                self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_last.pt'))
                self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_last.pt'))
                
                if avg_val_acc > best_avg_val_acc:
                    best_avg_val_acc = avg_val_acc
                    self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_best.pt'))
                    self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_best.pt'))
                
            tb_writer.add_scalars(f'losses', dict(index=index_loss, index_l1=index_l1_loss, total=total_loss), iter_index)
            pbar.set_description(f'loss: {total_loss:.6f}, index_loss: {index_loss:.6f}, index_l1_loss: {index_l1_loss:.6f}')
            if (iter_index + 1) >= hyps['val_freq']:
                tb_writer.add_scalars(f'accs/val_accs', val_accs, iter_index)
                tb_writer.add_scalar(f'accs/avg_val_acc', avg_val_acc, iter_index)
                pbar.set_description(f'loss: {total_loss:.6f}, index_loss: {index_loss:.6f}, index_l1_loss: {index_l1_loss:.6f}, '
                                     f'avg_val_acc: {avg_val_acc:.4f}')
            