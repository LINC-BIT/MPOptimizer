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


class ElasticDNN_MDPretrainingIndexAlg(BaseAlg):
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
            
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int,            
            'optimizer': str,
            'optimizer_args': dict,
            'index_optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'num_iters': int,
            'val_freq': int,
            'max_sparsity': float,
            'min_sparsity': float,
            'l1_reg_loss_weight': float,
            'index_loss_weight': float,
            'val_num_sparsities': int,
            
            'bn_cal_num_iters': int,
            
            'index_1_to_k': int
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
    
    def two_params_partial_diff(self, trained_p: torch.Tensor, ref_p: torch.Tensor, 
                             index):

        assert trained_p.dim() == ref_p.dim()
        # assert index.size(0) == trained_p.size(0) and index.size(1) == ref_p.size(0)
        
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
                
        elif trained_p.dim() == 1:
            trained_p = trained_p.unsqueeze(1)
            ref_p = ref_p.unsqueeze(1)
            
            # index = index.unsqueeze(-1)
        #     linear_combed_ref_p = (ref_p.unsqueeze(0) * index).sum(1)
        # else:
        
        # res = 0.
        # # slow version
        # for g1i, (g1, (_index, selected_g2is)) in enumerate(zip(trained_p, index)):
        #     comb_ref_p = (ref_p[torch.tensor(selected_g2is)] * _index.unsqueeze(1)).sum(0)
        #     res += ((comb_ref_p - g1) ** 2).sum()
        # return res
        # -------------
        
        # (train_p.size(0), 2, ref_p.size(1))
        # if trained_p.dim() == 2:
        
        # NOTE: fast version?
        selected_ref_p = torch.stack([ref_p[torch.tensor(selected_g2is)] for _, selected_g2is in index])
        # (train_p.size(0), 2)
        indexes = torch.stack([_index for _index, _ in index])
        
        # print(trained_p.size(), ref_p.size(), selected_ref_p.size(), indexes.size())
        
        # (train_p.size(),)
        # should be (train_p.size(0), ref_p.size(1))
        linear_combed_ref_p = (selected_ref_p * indexes.unsqueeze(-1)).sum(1)
        # -------------
            
        # else:
        #     selected_ref_p = torch.stack([ref_p[torch.tensor(selected_g2is)] for _, selected_g2is in index])
        #     # (train_p.size(0), 2)
        #     indexes = torch.stack([_index for _index, _ in index])
            
        #     # print(trained_p.size(), ref_p.size(), selected_ref_p.size(), indexes.size())
            
        #     # (train_p.size(),)
        #     # should be (train_p.size(0), ref_p.size(1))
        #     linear_combed_ref_p = (selected_ref_p * indexes).sum(1)
            
        diff = (linear_combed_ref_p - trained_p).norm(2) ** 2
        return diff
        
        # print(trained_p.size(), ref_p.size(), index.size())
        
        # if split_size is None:
        #     # old version: huge memory consumption, not recommended (although this is fastest)
        #     # print('old version')
        #     linear_combed_ref_p = (ref_p.unsqueeze(0) * index).sum(1)
        
        # else:
        #     # new version
        #     linear_combed_ref_p = 0
            
        #     cur_split_size = split_size
        #     while index.size(1) % cur_split_size != 0:
        #         cur_split_size -= 1
        #     # print(cur_split_size) 
            
        #     for i in range(0, index.size(1), cur_split_size):
        #         # if not isinstance(linear_combed_ref_p, int):
        #             # print(linear_combed_ref_p.size(), ref_p.unsqueeze(0)[:, i: i + cur_split_size].size(), index[:, i: i + cur_split_size].size())
        #         linear_combed_ref_p += ref_p.unsqueeze(0)[:, i: i + cur_split_size] * index[:, i: i + cur_split_size]
        #     linear_combed_ref_p = linear_combed_ref_p.sum(1)
            
        # diff = (linear_combed_ref_p - trained_p).norm(2) ** 2
        # return diff
        
    def get_index_loss(self, fm, md, indexes, match_fn):
        res = 0.

        for name, p in md.named_parameters():
            if p.dim() == 0:
                continue
            
            raw_p = match_fn(name, fm)
            if raw_p is None:
                continue

            index = indexes[name]
            
            # print(name)
            res += self.two_params_partial_diff(p, raw_p, index)
        return res
        
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
        
        # logger.info(f'init master DNN acc w/ FBS (before constructing indexes): {self.models["md"].get_accuracy(val_loader):.4f}')
        
        master_dnn = self.models['md'].models_dict['main']
        elastic_dnn_util = self.models['fm'].get_elastic_dnn_util()
        
        with torch.no_grad():
            indexes = {}
            trained_indexes = []
            for name, p in self.models['md'].models_dict['main'].named_parameters():
                if p.dim() > 0:
                    matched_p_in_fm = self.models['md'].get_matched_param_of_fm(name, self.models['fm'].models_dict['main'])
                    if matched_p_in_fm is None:
                        continue
                    
                    if p.dim() > 1:
                        p = p.flatten(1)
                        matched_p_in_fm = matched_p_in_fm.flatten(1)
                    
                        if p.size(1) < matched_p_in_fm.size(1):
                            p = self.upsample_2d_tensor(p, matched_p_in_fm.size(1))
                        
                    indexes[name] = []
                    for g1i, g1 in enumerate(p):
                        selected_g2is = torch.randperm(matched_p_in_fm.size(0))[0: hyps['index_1_to_k']]
                        index = torch.FloatTensor([1. / hyps['index_1_to_k']] * hyps['index_1_to_k']).to(device)
                        indexes[name] += [(index, list(selected_g2is))]
                        trained_indexes += [index]
                        # for selected_g2i in selected_g2is:
                        #     indexes[name][g1i][selected_g2i] = torch.FloatTensor([1. / hyps['index_1_to_k']]).to(device)
                        #     indexes[name][g1i][selected_g2i].requires_grad = True
                        #     indexes[name][g1i][selected_g2i] = 1.
                        #     trained_indexes += [indexes[name][g1i][selected_g2i]]
                            
                    # print(p.size(), selected_g2is)
                
                # NOTE: only constructing indexes between similar [weight row/filter] pair
                # for g1i, g1 in tqdm.tqdm(enumerate(p), dynamic_ncols=True, leave=False, total=len(p)):
                #     indexes[name][g1i] = {}
                    
                #     # similarities = []
                #     # for g2i, g2 in enumerate(matched_p_in_fm):
                #     #     similarity = ((g1 - g2) ** 2).sum()
                #     #     similarities += [similarity]
                #     if p.dim() == 1:
                #         similarities = ((g1 - matched_p_in_fm) ** 2)
                #     else:
                #         similarities = ((g1.unsqueeze(0) - matched_p_in_fm) ** 2).sum(1)
                #         assert similarities.size(0) == matched_p_in_fm.size(0)
                    
                #     most_similar_g2is = similarities.argsort(descending=True)
                #     accu_similarities = similarities.sort(descending=True)[0].cumsum(0)
                    
                #     for ai, accu_sim in enumerate(accu_similarities):
                #         if accu_sim > accu_similarities[-1] * hyps['index_construct_r']:
                #             break
                    
                #     # selected fm weight rows for constructing indexes
                #     selected_g2is = most_similar_g2is[0: ai]
                #     for selected_g2i in selected_g2is:
                #         indexes[name][g1i][selected_g2i] = torch.FloatTensor([1. / ai]).to(device)
                #         indexes[name][g1i][selected_g2i].requires_grad = True
                #         trained_indexes += [indexes[name][g1i][selected_g2i]]

                #         num_indexes += 1
                
                # index_percent = num_indexes / (matched_p_in_fm.size(0) * p.size(0)) * 100.
                # logger.info(f'layer {name}: constructing {index_percent:.3f}% of indexes')
                
        tmp_indexes_file_path = os.path.join(self.res_save_dir, 'tmp-indexes.pt')
        torch.save(indexes, tmp_indexes_file_path)
        logger.info(f'# indexes: {len(trained_indexes)}; generate indexes ({(os.path.getsize(tmp_indexes_file_path) / 1024**2):.3f}MB)')
        os.remove(tmp_indexes_file_path)
        
        # 2.1 train whole master DNN (knowledge distillation)
        for p in master_dnn.parameters():
            p.requires_grad = True
        self.models['md'].to_train_mode()
        for p in trained_indexes:
            p.requires_grad = True
        for p in self.models['fm'].models_dict['main'].parameters():
            p.requires_grad = False
        
        optimizer = torch.optim.__dict__[hyps['optimizer']]([
            {'params': self.models['md'].models_dict['main'].parameters(), **hyps['optimizer_args']},
            # {'params': trained_indexes, **hyps['index_optimizer_args']}
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
            index_loss = hyps['index_loss_weight'] * self.get_index_loss(self.models['fm'].models_dict['main'], 
                                                                         self.models['md'].models_dict['main'], 
                                                                         indexes,
                                                                         self.models['md'].get_matched_param_of_fm)
            total_loss = task_loss + l1_reg_loss + index_loss
            
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
                    self.models['md'].models_dict['main'] = md_for_test
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
                
            tb_writer.add_scalars(f'losses', dict(task=task_loss, l1_reg=l1_reg_loss, index=index_loss, total=total_loss), iter_index)
            pbar.set_description(f'loss: {total_loss:.6f}, task_loss: {task_loss:.6f}, index_loss: {index_loss:.6f}, '
                                 f'l1_loss: {l1_reg_loss:.6f}')
            if (iter_index + 1) >= hyps['val_freq']:
                tb_writer.add_scalars(f'accs/val_accs', val_accs, iter_index)
                tb_writer.add_scalar(f'accs/avg_val_acc', avg_val_acc, iter_index)
                pbar.set_description(f'loss: {total_loss:.6f}, task_loss: {task_loss:.6f}, index_loss: {index_loss:.6f}, '
                                     f'l1_loss: {l1_reg_loss:.6f}, avg_val_acc: {avg_val_acc:.4f}')
            