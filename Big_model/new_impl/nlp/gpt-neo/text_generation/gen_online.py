import os
gpt_neo_series_id = '1.3B_ckpt'
os.environ['gpt_neo_series_id'] = gpt_neo_series_id
import torch
import torch.nn as nn
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from gpt_neo import getTokenizer, ElasticGPTUtil, FMLoRA_GPT_Util, ElasticDNN_OfflineTextGenFMModel, ElasticDNN_OfflineTextGenMDModel, FM_to_MD_GPT_Util, collate_fn
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from methods.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md import ElasticDNN_MDPretrainingIndexAlg
from utils.dl.common.model import LayerActivation2, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_gen_scenario
import torch.nn.functional as F
import os
from utils.dl.common.loss import CrossEntropyLossSoft
from new_impl.cv.feat_align.main_gpt_neo import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from new_impl.cv.feat_align.mmd import mmd_rbf
from new_impl.cv.utils.elasticfm_da import init_online_model, elasticfm_da
from methods.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel
from utils.common.log import logger
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import json

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

device = 'cuda:1'
app_name = 'cls'
sd_sparsity = 0.8

settings = {
    'involve_fm': True
}

torch.cuda.set_device(1)

scenario = build_gen_scenario(
        source_datasets_name=['No_robots'],
        target_datasets_order=['Medicine_task', 'Law_task'] * 10,
        da_mode='close_set',
        data_dirs={
            'No_robots': '/data/zql/datasets/no_robots',
            'Law_task': '/data/zql/datasets/law_task',
            'Medicine_task': '/data/zql/datasets/medicine_task',
        },
    )

class ElasticDNN_TxtgenOnlineModel(ElasticDNN_OnlineModel):
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        tokenizer = getTokenizer()
        self.to_eval_mode()
        pred_txt = []
        true_txt = []
        res = []
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, _) in pbar:
                if len(x) == 0:
                    continue
                # if batch_index > 10:
                #     break
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                # input_ids = []
                inputlen = x['len']
                y = x['labels']
                x['labels'] = None
                outputs = self.models_dict['main'].generate(x, pad_id=tokenizer.pad_token_id)
                
                for i, op in enumerate(outputs):
                    op = op.tolist()
                    op = list(filter(lambda x: x != tokenizer.pad_token_id, op))
                    txt = tokenizer.decode(op)
                    txt = txt.replace(tokenizer.pad_token, "")
                    res.append(txt)
                    txt = txt[inputlen[i]:]
                    pred_txt.append(nltk.word_tokenize(txt))
                for tp in y:
                    true_txt.append(nltk.word_tokenize(tokenizer.decode(tp).replace(tokenizer.pad_token, '')))
                # pred = F.softmax(output, dim=1).argmax(dim=1)
                # correct = torch.eq(pred, y).sum().item()
                # acc += correct
                sample_num += len(y)
                
                # pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                #                      f'cur_batch_acc: {(correct / len(y)):.4f}')
        json.dump(res, open("./gpt_generation.json", "w"))
        smooth = SmoothingFunction()
        score = 0.
        for pred, true in zip(pred_txt, true_txt):
            score += sentence_bleu([true], pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        score /= sample_num
        return score
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticGPTUtil()
    
    def get_fm_matched_param_of_md_param(self, md_param_name):
        # only between qkv.weight, norm.weight/bias
        self_param_name = md_param_name
        fm = self.models_dict['fm']
        # if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
        #     return None
        
        # p = get_parameter(self.models_dict['md'], self_param_name)
        # if p.dim() == 0:
        #     return None
        # elif p.dim() == 1 and 'norm' in self_param_name and 'weight' in self_param_name:
        #     return get_parameter(fm, self_param_name)
        
        if any([k in self_param_name for k in ['fbs', 'wte', 'wpe']]):
            return None
        
        p = get_parameter(self.models_dict['md'], self_param_name)
        if p.dim() == 0:
            return None
        # elif p.dim() == 1 and 'layernorm' in self_param_name and 'weight' in self_param_name:
        #     return get_parameter(fm, self_param_name)
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        # if 'qkv.weight' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -1]) + '.qkv'
        #     fm_qkv = get_module(fm, fm_qkv_name)
            
        #     fm_abs_name = '.'.join(ss[0: -1]) + '.abs'
        #     fm_abs = get_module(fm, fm_abs_name)
            
        #     # NOTE: unrecoverable operation! multiply LoRA parameters to allow it being updated in update_fm_param()
        #     # TODO: if fm will be used for inference, _mul_lora_weight will not be applied!
        #     if not hasattr(fm_abs, '_mul_lora_weight'):
        #         logger.debug(f'set _mul_lora_weight in {fm_abs_name}')
        #         setattr(fm_abs, '_mul_lora_weight', 
        #                 nn.Parameter(torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0)))
            
        #     return torch.cat([
        #         fm_qkv.weight.data, # task-agnositc params
        #         fm_abs._mul_lora_weight.data # task-specific params (LoRA)
        #     ], dim=0)
            
        # # elif 'to_qkv.bias' in self_param_name:
        # #     ss = self_param_name.split('.')
            
        # #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        # #     return get_parameter(fm, fm_qkv_name)
            
        # elif 'mlp.fc1' in self_param_name and 'weight' in self_param_name:
        #     fm_param_name = self_param_name.replace('.linear', '')
        #     return get_parameter(fm, fm_param_name)

        # elif 'mlp.fc2' in self_param_name and 'weight' in self_param_name:
        #     fm_param_name = self_param_name
        #     return get_parameter(fm, fm_param_name)
        
        # else:
        #     # return get_parameter(fm, self_param_name)
        #     return None
        if ('q_proj' in self_param_name or 'k_proj' in self_param_name or \
        'v_proj' in self_param_name) and ('weight' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            if not hasattr(fm_abs, '_mul_lora_weight'):
                logger.debug(f'set _mul_lora_weight in {fm_abs_name}')
                setattr(fm_abs, '_mul_lora_weight', 
                        nn.Parameter(fm_abs[1].weight @ fm_abs[0].weight))
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs._mul_lora_weight.data # task-specific params (LoRA)
            ], dim=0)
            
        elif ('q_proj' in self_param_name or 'k_proj' in self_param_name or \
        'v_proj' in self_param_name) and ('bias' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc.bias'
            return get_parameter(fm, fm_qkv_name)
        
        elif 'mlp.c_fc' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)
        elif 'mlp.c_fc' in self_param_name and 'bias' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)
        # elif 'mlp.fc2' in self_param_name and 'weight' in self_param_name:
        #     fm_param_name = self_param_name.replace('.linear', '')
        #     return get_parameter(fm, fm_param_name)
        else:
            #return get_parameter(fm, self_param_name)
            return None


    def update_fm_param(self, md_param_name, cal_new_fm_param_by_md_param):
        if not (('q_proj' in md_param_name or 'k_proj' in md_param_name or \
        'v_proj' in md_param_name) and ('weight' in md_param_name)):
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
        if any([k in self_param_name for k in ['fbs', 'wte', 'wpe']]):
            return None
        
        p = get_parameter(self.models_dict['sd'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1 and ('LayerNorm' in self_param_name or 'ln' in self_param_name) and 'weight' in self_param_name:
            return get_parameter(md, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if ('q_proj' in self_param_name or 'k_proj' in self_param_name or \
        'v_proj' in self_param_name) and ('weight' in self_param_name):       
            return get_parameter(md, self_param_name) # NOTE: no fbs in qkv!
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
            
        elif 'mlp.c_fc.0.weight' in self_param_name:
            fm_param_name = '.'.join(self_param_name.split('.')[0: -2]) + '.linear.weight'
            return get_parameter(md, fm_param_name)

        elif 'mlp.c_fc.0.bias' in self_param_name:
            fm_param_name = '.'.join(self_param_name.split('.')[0: -2]) + '.linear.bias'
            return get_parameter(md, fm_param_name)

        elif 'mlp.c_proj' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name
            return get_parameter(md, fm_param_name)
        
        elif 'static_channel_attention' not in self_param_name:
            return get_parameter(md, self_param_name)
            # return None
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['sd'], 'classifier')
        return list(head.parameters())
    
    
    
class TxtgenOnlineFeatAlignModel(OnlineFeatAlignModel):
    def get_trained_params(self): # TODO: elastic fm only train a part of params
        #qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters() if 'attention.attention.projection_query' in n or 'attention.attention.projection_key' in n or 'attention.attention.projection_value' in n or 'intermediate.dense' in n or 'output.dense' in n]
        qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters()]
        return qkv_and_norm_params
    
    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'model.lm_head'))
    
    def forward_to_get_task_loss(self, x, y):
        losses = self.infer(x)
        # print(losses)
        
        return losses
    
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
        acc = 0
        sample_num = 0
        tokenizer = getTokenizer()
        self.to_eval_mode()
        pred_txt = []
        true_txt = []
        res = []
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, _) in pbar:
                if len(x) == 0:
                    continue
                # if batch_index > 10:
                #     break
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                # input_ids = []
                inputlen = x['len']
                y = x['labels']
                x['labels'] = None
                outputs = self.models_dict['main'].generate(x, pad_id=tokenizer.pad_token_id)
                
                for i, op in enumerate(outputs):
                    op = op.tolist()
                    op = list(filter(lambda x: x != tokenizer.pad_token_id, op))
                    txt = tokenizer.decode(op)
                    txt = txt.replace(tokenizer.pad_token, "")
                    res.append(txt)
                    txt = txt[inputlen[i]:]
                    pred_txt.append(nltk.word_tokenize(txt))
                for tp in y:
                    true_txt.append(nltk.word_tokenize(tokenizer.decode(tp).replace(tokenizer.pad_token, '')))
                # pred = F.softmax(output, dim=1).argmax(dim=1)
                # correct = torch.eq(pred, y).sum().item()
                # acc += correct
                sample_num += len(y)
                
                # pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                #                      f'cur_batch_acc: {(correct / len(y)):.4f}')
        json.dump(res, open("./gpt_generation.json", "w"))
        smooth = SmoothingFunction()
        score = 0.
        for pred, true in zip(pred_txt, true_txt):
            score += sentence_bleu([true], pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        score /= sample_num
        return score




#from new_impl.cv.model import ElasticDNN_ClsOnlineModel
elasticfm_model = ElasticDNN_TxtgenOnlineModel('gen', init_online_model(
    'new_impl/nlp/gpt-neo/text_generation/results/gen_md_w_fbs_index.py/20231222/999995-003118-results/models/fm_best.pt',
    'new_impl/nlp/gpt-neo/text_generation/results/gen_md_w_fbs_index.py/20231222/999995-003118-results/models/md_best.pt',
    'gen', __file__
), device, {
    'md_to_fm_alpha': 0.01,
    'fm_to_md_alpha': 0.1
})

da_alg = FeatAlignAlg
from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
#from new_impl.cv.model import ClsOnlineFeatAlignModel
da_model = TxtgenOnlineFeatAlignModel
da_alg_hyp = {
    'Medicine_task': {
        'train_batch_size': 2,
        'val_batch_size': 1,
        'num_workers': 2,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 1000,
        'val_freq': 200,
        'sd_sparsity':0.3,
        'feat_align_loss_weight': 1.0,
    },
    'Law_task': {
        'train_batch_size': 2,
        'val_batch_size': 1,
        'num_workers': 2,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 1000,
        'val_freq': 200,
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
    collate_fn=collate_fn
)