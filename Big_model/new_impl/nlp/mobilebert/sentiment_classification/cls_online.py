from typing import Any, Dict, List
from schema import Schema
from data import build_scenario, build_cl_scenario, CLScenario, MergedDataset
from data.dataloader import build_dataloader
# from methods.elasticdnn.api.online_model import ElasticDNN_OnlineModel
from methods.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.gem.gem_el_bert import OnlineGEMModel, GEMAlg
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from copy import deepcopy
from bert import ElasticBertUtil
import torch
import torch.nn.functional as F
import sys
import tqdm
from torch import nn
from utils.common.log import logger

# from methods.shot.shot import OnlineShotModel
from experiments.utils.elasticfm_cl import init_online_model, elasticfm_cl
# torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cuda'
app_name = 'secls'
sd_sparsity = 0.8

class ElasticDNN_SeClsOnlineModel(ElasticDNN_OnlineModel):
    
    @torch.no_grad()
    def sd_feedback_to_md(self, after_da_sd, unpruned_indexes_of_layers):
        self.models_dict['sd'] = after_da_sd
        self.before_da_md = deepcopy(self.models_dict['md'])
        
        logger.info('\n\nsurrogate DNN feedback to master DNN...\n\n')
        # one-to-one
        
        cur_unpruned_indexes = None
        cur_unpruned_indexes_name = None
        
        for p_name, p in self.models_dict['sd'].named_parameters():
            matched_md_param = self.get_md_matched_param_of_sd_param(p_name)
            logger.debug(f'if feedback: {p_name}')
            if matched_md_param is None:
                continue
            logger.debug(f'start feedback: {p_name}, {p.size()} -> {matched_md_param.size()}')
            # average
            # setattr(matched_md_module, matched_md_param_name, (matched_md_param + p) / 2.)
            
            if p_name in unpruned_indexes_of_layers.keys():
                cur_unpruned_indexes = unpruned_indexes_of_layers[p_name]
                cur_unpruned_indexes_name = p_name
            
            if p.size() != matched_md_param.size():
                logger.debug(f'cur unpruned indexes: {cur_unpruned_indexes_name}, {cur_unpruned_indexes.size()}')
                
                if p.dim() == 1: # norm
                    new_p = deepcopy(matched_md_param)
                    new_p[cur_unpruned_indexes] = p
                elif p.dim() == 2: # linear
                    if p.size(0) < matched_md_param.size(0): # output pruned
                        new_p = deepcopy(matched_md_param)
                        new_p[cur_unpruned_indexes] = p
                    else: # input pruned
                        new_p = deepcopy(matched_md_param)
                        new_p[:, cur_unpruned_indexes] = p
                p = new_p
                
            assert p.size() == matched_md_param.size(), f'{p.size()}, {matched_md_param.size()}'
            
            if 'classifier' in p_name:
                continue
            # if False:
                # self.last_trained_cls_indexes 
                assert hasattr(self, 'last_trained_cls_indexes')
                print(self.last_trained_cls_indexes)

                diff = self._compute_diff(matched_md_param, p)
                # matched_md_param[self.last_trained_cls_indexes].copy_(p[self.last_trained_cls_indexes.to(self.device)])
                matched_md_param.copy_(p)
                logger.debug(f'SPECIFIC FOR CL HEAD | end feedback: {p_name}, diff: {diff:.6f}')
            else:
                diff = self._compute_diff(matched_md_param, (matched_md_param + p) / 2.)
                matched_md_param.copy_((matched_md_param + p) / 2.)
                logger.debug(f'end feedback: {p_name}, diff: {diff:.6f}')
            
    def add_cls_in_head(self, num_cls):
        head: nn.Linear = get_module(self.models_dict['md'], 'classifier')
        
        new_head = nn.Linear(head.in_features, head.out_features + num_cls, head.bias is not None, device=self.device)
        
        # nn.init.zeros_(new_head.weight.data)
        # nn.init.zeros_(new_head.bias.data)
        
        new_head.weight.data[0: head.out_features] = deepcopy(head.weight.data)
        new_head.bias.data[0: head.out_features] = deepcopy(head.bias.data)
        set_module(self.models_dict['md'], 'classifier', new_head)
        set_module(self.models_dict['fm'], 'classifier', new_head)
        
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                y = y.to(self.device)
                output = self.infer(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticBertUtil()
    
    def get_fm_matched_param_of_md_param(self, md_param_name):
        # only between qkv.weight, norm.weight/bias
        self_param_name = md_param_name
        fm = self.models_dict['fm']
        if any([k in self_param_name for k in ['fbs', 'ab', 'embeddings','ln']]):
            return None
        
        p = get_parameter(self.models_dict['md'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1 and 'LayerNorm' in self_param_name and 'weight' in self_param_name:
            return get_parameter(fm, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('weight' in self_param_name):
            
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            # NOTE: unrecoverable operation! multiply LoRA parameters to allow it being updated in update_fm_param()
            # TODO: if fm will be used for inference, _mul_lora_weight will not be applied!
            if not hasattr(fm_abs, '_mul_lora_weight'):
                logger.debug(f'set _mul_lora_weight in {fm_abs_name}')
                setattr(fm_abs, '_mul_lora_weight', 
                        nn.Parameter(fm_abs[1].weight @ fm_abs[0].weight))
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs._mul_lora_weight.data # task-specific params (LoRA)
            ], dim=0)
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
            
        elif 'dense' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)

        # elif 'mlp.fc2' in self_param_name and 'weight' in self_param_name:
        #     fm_param_name = self_param_name
        #     return get_parameter(fm, fm_param_name)
        
        else:
            # return get_parameter(fm, self_param_name)
            return None
        
    def update_fm_param(self, md_param_name, cal_new_fm_param_by_md_param):
        if not ('query' in md_param_name or 'key' in md_param_name or 'value' in md_param_name):
            matched_fm_param_ref = self.get_fm_matched_param_of_md_param(md_param_name)
            matched_fm_param_ref.copy_(cal_new_fm_param_by_md_param)
        else:
            new_fm_attn_weight, new_fm_lora_weight = torch.chunk(cal_new_fm_param_by_md_param, 2, 0)
            
            ss = md_param_name.split('.')
            fm = self.models_dict['fm']
            
            # update task-agnostic parameters
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            fm_qkv.weight.data.copy_(new_fm_attn_weight)
            
            # update task-specific parameters
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            fm_abs._mul_lora_weight.data.copy_(new_fm_lora_weight) # TODO: this will not be applied in inference!
        
    def get_md_matched_param_of_fm_param(self, fm_param_name):
        return super().get_md_matched_param_of_fm_param(fm_param_name)
    
    def get_md_matched_param_of_sd_param(self, sd_param_name):
        # raise NotImplementedError

        # only between qkv.weight, norm.weight/bias
        self_param_name = sd_param_name
        md = self.models_dict['md']
        if any([k in self_param_name for k in ['fbs', 'ab', 'embeddings']]):
            return None
        
        p = get_parameter(self.models_dict['sd'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1 and 'LayerNorm' in self_param_name and 'weight' in self_param_name:
            return get_parameter(md, self_param_name)
        
        if 'classifier' in self_param_name:
            return get_parameter(md, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('weight' in self_param_name):
            
        
            return get_parameter(md, self_param_name) # NOTE: no fbs in qkv!
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
            
        elif 'intermediate.dense.0.weight' in self_param_name:
            fm_param_name = '.'.join(self_param_name.split('.')[0: -2]) + '.linear.weight'
            return get_parameter(md, fm_param_name)

        elif 'output.dense' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name
            return get_parameter(md, fm_param_name)
        
        else:
            # return get_parameter(fm, self_param_name)
            return None
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['sd'], 'classifier')
        return list(head.parameters())

class SeClsOnlineGEMModel(OnlineGEMModel):
    def get_trained_params(self):
        qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters() if 'query' in n or 'key' in n or 'value' in n or 'dense' in n or 'LayerNorm' in n]
        return qkv_and_norm_params
    
    def forward_to_get_task_loss(self, x, y):
        return F.cross_entropy(self.infer(x), y)
    
    def add_cls_in_head(self, num_cls):
        return

        head: nn.Linear = get_module(self.models_dict['main'], 'head')
        
        new_head = nn.Linear(head.in_features, head.out_features + num_cls, head.bias is not None, device=self.device)
        new_head.weight.data[0: head.out_features] = deepcopy(head.weight.data)
        new_head.bias.data[0: head.out_features] = deepcopy(head.bias.data)
        set_module(self.models_dict['main'], 'head', new_head)
        
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](**x)
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        _d = test_loader.dataset
        from data import build_dataloader, split_dataset
        if _d.__class__.__name__ == '_SplitDataset' and _d.underlying_dataset.__class__.__name__ == 'MergedDataset': # necessary for CL
            print('\neval on merged datasets')
            
            merged_full_dataset = _d.underlying_dataset.datasets
            ratio = len(_d.keys) / len(_d.underlying_dataset)
            
            if int(len(_d) * ratio) == 0:
                ratio = 1.
            # print(ratio)
            # bs = 
            # test_loaders = [build_dataloader(split_dataset(d, min(max(test_loader.batch_size, int(len(d) * ratio)), len(d)))[0], # TODO: this might be overlapped with train dataset
            #                                  min(test_loader.batch_size, int(len(d) * ratio)), 
            #                                  test_loader.num_workers, False, None) for d in merged_full_dataset]

            test_loaders = []
            for d in merged_full_dataset:
                n = int(len(d) * ratio)
                if n == 0:
                    n = len(d)
                sub_dataset = split_dataset(d, min(max(test_loader.batch_size, n), len(d)))[0]
                loader = build_dataloader(sub_dataset, min(test_loader.batch_size, n), test_loader.num_workers, False, None)
                test_loaders += [loader]
            
            accs = [self.get_accuracy(loader) for loader in test_loaders]
            print(accs)
            return sum(accs) / len(accs)
            
            
            
        
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                y = y.to(self.device)
                output = self.infer(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                # if batch_index == 0:
                #     print(pred, y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc

settings = {
    'involve_fm': True
}

scenario = build_scenario(
    source_datasets_name=['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                            'HL5Domains-NikonCoolpix4300'],
    target_datasets_order=['HL5Domains-Nokia6610','Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker', 
                           'Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
                           'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
                           'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100', 
                           'SemEval-Laptop', 'SemEval-Rest'] * 2 + ['Liu3Domains-Computer', 'Liu3Domains-Router'],
    da_mode='close_set',
    data_dirs={
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
            for k in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                            'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing3Domains/asc/{k.split("-")[1]}' 
            for k in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker']},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing9Domains/asc/{k.split("-")[1]}' 
            for k in ['Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
                           'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
                           'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100']},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/XuSemEval/asc/14/{k.split("-")[1].lower()}' 
            for k in ['SemEval-Laptop', 'SemEval-Rest']},
    },
)

scenario = build_cl_scenario(
    da_scenario=scenario,
    target_datasets_name=['HL5Domains-Nokia6610'] * 16,
    num_classes_per_task=5,
    max_num_tasks=16,
    data_dirs={
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
            for k in ['HL5Domains-Nokia6610']}
    }
)

elasticfm_model = ElasticDNN_SeClsOnlineModel('secls', init_online_model(
    'new_impl/nlp/mobilebert/sentiment_classification/results/cls_md_w_fbs_index.py/20231019/999999-222456-result/models/fm_best.pt',
    'new_impl/nlp/mobilebert/sentiment_classification/results/cls_md_w_fbs_index.py/20231019/999999-222456-result/models/md_best.pt',
    'cls', __file__
), device, {
    'md_to_fm_alpha': 0.2,
    'fm_to_md_alpha': 0.2
})

da_alg = GEMAlg
da_model = SeClsOnlineGEMModel
da_alg_hyp = {
    'train_batch_size': 4,
    'val_batch_size': 16,
    'num_workers': 4,
    'optimizer': 'AdamW',
    'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
    'scheduler': '',
    'scheduler_args': {},
    'num_iters': 100,
    'val_freq': 20,
    'n_memories': 4 ,
    'n_inputs': 3 * 224 * 224,
    'margin': 0.5,
    'num_my_iters': 0,
    'sd_sparsity': 0.7
}


elasticfm_cl(
    [app_name],
    [scenario],
    [elasticfm_model],
    [da_alg],
    [da_alg_hyp],
    [da_model],
    device,
    settings,
    __file__,
    "results"
)