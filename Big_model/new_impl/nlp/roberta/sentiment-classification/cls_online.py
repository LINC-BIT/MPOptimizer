import os
#bert_path should be the path of the roberta-base dir
os.environ['bert_path'] = '/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/nlp/roberta/sentiment-classification/roberta-base'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from methods.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md import ElasticDNN_MDPretrainingIndexAlg
from utils.dl.common.model import LayerActivation2, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
import torch.nn.functional as F
from utils.dl.common.loss import CrossEntropyLossSoft
from new_impl.cv.feat_align.main import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from new_impl.cv.feat_align.mmd import mmd_rbf
from new_impl.cv.utils.elasticfm_da import init_online_model, elasticfm_da
from methods.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel
from utils.common.log import logger
import json
from roberta import FMLoRA_Roberta_Util, RobertaForSenCls, FM_to_MD_Roberta_Util, ElasticRobertaUtil
from copy import deepcopy

torch.cuda.set_device(1)

# from methods.shot.shot import OnlineShotModel
from experiments.utils.elasticfm_cl import init_online_model, elasticfm_cl
# torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cuda:1'
app_name = 'secls'
sd_sparsity = 0.8

settings = {
    'involve_fm': True
}

scenario = build_scenario(
        source_datasets_name=['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB'],
        target_datasets_order=['HL5Domains-Nokia6610', 'HL5Domains-NikonCoolpix4300'] * 10, # TODO
        da_mode='close_set',
        data_dirs={
            **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
             for k in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                              'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']}
        },
    )

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
            
            # if 'classifier' in p_name:
            #     continue
            # # if False:
            #     # self.last_trained_cls_indexes 
            #     assert hasattr(self, 'last_trained_cls_indexes')
            #     print(self.last_trained_cls_indexes)

            #     diff = self._compute_diff(matched_md_param, p)
            #     # matched_md_param[self.last_trained_cls_indexes].copy_(p[self.last_trained_cls_indexes.to(self.device)])
            #     matched_md_param.copy_(p)
            #     logger.debug(f'SPECIFIC FOR CL HEAD | end feedback: {p_name}, diff: {diff:.6f}')
            # else:
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
        return ElasticRobertaUtil()
    
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

        elif ('query' in self_param_name or 'key' in self_param_name or 'value' in self_param_name) \
           and 'bias' in self_param_name:
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc.bias'
            return get_parameter(fm, fm_qkv_name)

        elif 'intermediate.dense' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)

        else:
            return get_parameter(fm, self_param_name)
        
    def update_fm_param(self, md_param_name, cal_new_fm_param_by_md_param):
        if not ('query' in md_param_name or 'key' in md_param_name or 'value' in md_param_name):
            matched_fm_param_ref = self.get_fm_matched_param_of_md_param(md_param_name)
            matched_fm_param_ref.copy_(cal_new_fm_param_by_md_param)
        elif 'bias' in md_param_name:
            ss = md_param_name.split('.')
            fm = self.models_dict['fm']

            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)

            fm_qkv.bias.data.copy_(cal_new_fm_param_by_md_param)

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
        elif 'static_channel_attention' in self_param_name:
            return None


        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('weight' in self_param_name):
            
        
            return get_parameter(md, self_param_name) # NOTE: no fbs in qkv!
            
        elif ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('bias' in self_param_name):
            
        
            return get_parameter(md, self_param_name)
        elif 'intermediate.dense.0.weight' in self_param_name:
            fm_param_name = '.'.join(self_param_name.split('.')[0: -2]) + '.linear.weight'
            return get_parameter(md, fm_param_name)

        elif 'intermediate.dense.0.bias' in self_param_name:
            fm_param_name = '.'.join(self_param_name.split('.')[0: -2]) + '.linear.bias'
            return get_parameter(md, fm_param_name)

        elif 'output.dense' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name
            return get_parameter(md, fm_param_name)
        
        else:
            return get_parameter(md, self_param_name)
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['sd'], 'classifier')
        return list(head.parameters())

class SeClsOnlineFeatAlignModel(OnlineFeatAlignModel):
    def get_trained_params(self): # TODO: elastic fm only train a part of params
        #qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters() if 'attention.attention.projection_query' in n or 'attention.attention.projection_key' in n or 'attention.attention.projection_value' in n or 'intermediate.dense' in n or 'output.dense' in n]
        qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters()]
        return qkv_and_norm_params
    
    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'classifier'))
    
    def forward_to_get_task_loss(self, x, y):
        self.to_train_mode()
        return F.cross_entropy(self.infer(x), y)
    
    def get_mmd_loss(self, f1, f2):
        common_shape = min(f1.shape[0], f2.shape[0])
        f1 = f1.view(f1.shape[0], -1)
        f2 = f2.view(f2.shape[0], -1)
        f1 = f1[:common_shape,:]
        f2 = f2[:common_shape,:]
        return mmd_rbf(f1, f2)
    
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

elasticfm_model = ElasticDNN_SeClsOnlineModel('secls', init_online_model(
    'new_impl/nlp/roberta/sentiment-classification/results/cls_md_w_fbs_index.py/20240111/999998-203106-results/models/fm_best.pt',
    'new_impl/nlp/roberta/sentiment-classification/results/cls_md_w_fbs_index.py/20240111/999998-203106-results/models/md_best.pt',
    'cls', __file__
), device, {
    'md_to_fm_alpha': 0.01,
    'fm_to_md_alpha': 0.1
})

da_alg = FeatAlignAlg
from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
#from new_impl.cv.model import ClsOnlineFeatAlignModel
da_model = SeClsOnlineFeatAlignModel

da_alg_hyp = {
    'HL5Domains-Nokia6610': {
        'train_batch_size': 32,
        'val_batch_size': 256,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 2e-7, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'sd_sparsity':0.3,
        'feat_align_loss_weight': 1.0,
    },
    'HL5Domains-NikonCoolpix4300': {
        'train_batch_size': 32,
        'val_batch_size': 128,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 2e-7, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100, 
        'val_freq': 20,
        'sd_sparsity':0.3,
        'feat_align_loss_weight': 1.0,
    },
}


elasticfm_da(
    [app_name],
    [scenario],
    [elasticfm_model],
    [da_alg],
    [da_alg_hyp],
    [da_model],
    device,
    settings,
    __file__,
    "results",
)
