from typing import Any, Dict, List
from schema import Schema
from data import Scenario, MergedDataset
from methods.base.alg import BaseAlg
from methods.base.model import BaseModel
from data import build_dataloader
import torch.optim
import tqdm
import os
import time
from abc import abstractmethod
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
import torch.nn.functional as F

class OnlineFeatAlignModel(BaseModel):
    def get_required_model_components(self) -> List[str]:
        return ['main']
    
    @abstractmethod
    def get_feature_hook(self):
        pass
    
    @abstractmethod
    def forward_to_get_task_loss(self, x, y):
        pass

    @abstractmethod
    def get_trained_params(self):
        pass
    
    @abstractmethod
    def get_mmd_loss(self, f1, f2):
        pass
    

class FeatAlignAlg(BaseAlg):
    def get_required_models_schema(self) -> Schema:
        return Schema({
            'main': OnlineFeatAlignModel
        })
        
    def get_required_hyp_schema(self) -> Schema:
        from schema import Optional
        return Schema({
            'train_batch_size': int,
            'val_batch_size': int,
            'num_workers': int,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'num_iters': int,
            'val_freq': int,
            'feat_align_loss_weight': float,
            Optional('transform'): Compose,
        })
        
    def run(self, scenario: Scenario, hyps: Dict, collate_fn=None) -> Dict[str, Any]:
        super().run(scenario, hyps)
        
        assert isinstance(self.models['main'], OnlineFeatAlignModel) # for auto completion
        
        cur_domain_name = scenario.target_domains_order[scenario.cur_domain_index]
        if 'transform' in hyps.keys():
            datasets_for_training = scenario.get_online_cur_domain_datasets_for_training(transform=hyps['transform'])
        else:
            datasets_for_training = scenario.get_online_cur_domain_datasets_for_training()
        train_dataset = datasets_for_training[cur_domain_name]['train']

        if 'transform' in hyps.keys():
            datasets_for_inference = scenario.get_online_cur_domain_datasets_for_inference(transform=hyps['transform'])
        else:
            datasets_for_inference = scenario.get_online_cur_domain_datasets_for_inference()
        test_dataset = datasets_for_inference
        
        train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
                            True, None, collate_fn=collate_fn))
        # test_loader = build_dataloader(test_dataset, hyps['val_batch_size'], hyps['num_workers'],
        #                                False, False, collate_fn=collate_fn)
        
        source_datasets = [d['train'] for n, d in datasets_for_training.items() if n != cur_domain_name]
        source_dataset = MergedDataset(source_datasets)
        # source_dataset_val = [d['val'] for n, d in datasets_for_training.items() if n != cur_domain_name][0]
        source_train_loader = iter(build_dataloader(source_dataset, hyps['train_batch_size'], hyps['num_workers'],
                            True, None, collate_fn=collate_fn))
        # source_val_loader = build_dataloader(source_dataset_val, hyps['val_batch_size'], hyps['num_workers'],
        #                     False, False, collate_fn=collate_fn)
        # 1. generate surrogate DNN
        # for n, m in self.models['main'].models_dict['md'].named_modules():
        #     if isinstance(m, nn.Linear):
        #         m.reset_parameters()
        # from utils.dl.common.model import set_module
        
        # for n, m in self.models['main'].models_dict['md'].named_modules():
        #     if m.__class__.__name__ == 'KTakesAll':
        #         set_module(self.models['main'].models_dict['md'], n, KTakesAll(0.5))
        # self.models['main'].set_sd_sparsity(hyps['sd_sparsity'])
        device = self.models['main'].device
        # surrogate_dnn = self.models['main'].generate_sd_by_target_samples(next(train_loader)[0].to(device))
        # self.models['sd'] = surrogate_dnn
        
        # 2. train surrogate DNN
        # TODO: train only a part of filters
        trained_params = self.models['main'].get_trained_params()
        optimizer = torch.optim.__dict__[hyps['optimizer']](trained_params, **hyps['optimizer_args'])
        if hyps['scheduler'] != '':
            scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
        else:
            scheduler = None
        
        pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True, desc='da...')
        task_losses, mmd_losses = [], []
        accs = []
        
        total_train_time = 0.
        
        feature_hook = self.models['main'].get_feature_hook()
        
        num_test = 256

        for iter_index in pbar:
            
            if iter_index % hyps['val_freq'] == 0:
                from data import split_dataset
                if 'transform' in hyps.keys():
                    cur_test_batch_dataset = split_dataset(test_dataset, num_test, iter_index, transform=hyps['transform'])[0]
                else:
                    cur_test_batch_dataset = split_dataset(test_dataset, num_test, iter_index)[0]
                cur_test_batch_dataloader = build_dataloader(cur_test_batch_dataset, hyps['val_batch_size'], hyps['num_workers'], False, False, collate_fn=collate_fn)
                cur_acc = self.models['main'].get_accuracy(cur_test_batch_dataloader)
                # cur_source_acc = self.models['main'].get_accuracy(source_val_loader)
                accs += [{
                    'iter': iter_index,
                    'acc': cur_acc
                }]
            
            cur_start_time = time.time()
            
            self.models['main'].to_train_mode()
            
            x, _ = next(train_loader)
            
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
            else:
                x = x.to(device)
            
            source_x, source_y = next(source_train_loader)
            
            if isinstance(source_x, dict):
                for k, v in source_x.items():
                    if isinstance(v, torch.Tensor):
                        source_x[k] = v.to(device)
                source_y = source_y.to(device)
            else:
                source_x, source_y = source_x.to(device), source_y.to(device)
            
            task_loss = self.models['main'].forward_to_get_task_loss(source_x, source_y)
            source_features = feature_hook.input

            source_vis_features = source_features[1]
            tlist = []
            for vis in source_vis_features:
                tmp = F.max_pool2d(input=vis, kernel_size=2, stride=2, return_indices=False)
                tmp = tmp.view(vis.shape[0], -1)
                tlist.append(tmp)
            source_vis_features = torch.cat(tlist, dim=1)
            source_lang_features = source_features[3]
            source_lang_aggregate_feature = source_lang_features['aggregate']
            source_lang_embedded_feature = source_lang_features['embedded'].flatten(1)
            source_lang_hidden_feature = source_lang_features['hidden'].flatten(1)
            source_features = torch.cat([source_vis_features, source_lang_aggregate_feature, source_lang_embedded_feature, source_lang_hidden_feature], dim=1)
            self.models['main'].infer(x)
            target_features = feature_hook.input
            
            target_vis_features = target_features[1]
            tlist = []
            for vis in target_vis_features:
                tmp = F.max_pool2d(input=vis, kernel_size=2, stride=2, return_indices=False)
                tmp = tmp.view(vis.shape[0], -1)
                tlist.append(tmp)
            target_vis_features = torch.cat(tlist, dim=1)
            target_lang_features = target_features[3]
            target_lang_aggregate_feature = target_lang_features['aggregate']
            target_lang_embedded_feature = target_lang_features['embedded'].flatten(1)
            target_lang_hidden_feature = target_lang_features['hidden'].flatten(1)
            target_features = torch.cat([target_vis_features, target_lang_aggregate_feature, target_lang_hidden_feature, target_lang_hidden_feature], dim=1)

            mmd_loss = self.models['main'].get_mmd_loss(source_features, target_features)
        
            loss = task_loss + hyps['feat_align_loss_weight'] * mmd_loss
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            pbar.set_description(f'da... | cur_acc: {cur_acc:.4f}, task_loss: {task_loss:.6f}, mmd_loss: {mmd_loss:.6f}')
            task_losses += [float(task_loss.cpu().item())]
            mmd_losses += [float(mmd_loss.cpu().item())]
            
            total_train_time += time.time() - cur_start_time
        
        feature_hook.remove()
        
        time_usage = total_train_time
        
        plt.plot(task_losses, label='task')
        plt.plot(mmd_losses, label='mmd')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig(os.path.join(self.res_save_dir, 'loss.png'))
        plt.clf()

        if 'transform' in hyps.keys():
            cur_test_batch_dataset = split_dataset(test_dataset, num_test, iter_index + 1, transform=hyps['transform'])[0]
        else:
            cur_test_batch_dataset = split_dataset(test_dataset, num_test, iter_index + 1)[0]
        cur_test_batch_dataloader = build_dataloader(cur_test_batch_dataset, hyps['val_batch_size'], hyps['num_workers'], False, False, collate_fn=collate_fn)
        cur_acc = self.models['main'].get_accuracy(cur_test_batch_dataloader)
        accs += [{
            'iter': iter_index + 1,
            'acc': cur_acc
        }]
        
        return {
            'accs': accs,
            'time': time_usage
        }, self.models