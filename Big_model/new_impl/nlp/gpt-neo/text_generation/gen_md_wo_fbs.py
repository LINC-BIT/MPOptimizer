import os
gpt_neo_series_id = '1.3B_ckpt'
os.environ['gpt_neo_series_id'] = gpt_neo_series_id
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import torch
import torch.nn as nn
from methods.elasticdnn.api.model import ElasticDNN_OfflineSenClsFMModel, ElasticDNN_OfflineSenClsMDModel
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs_gpt_neo import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from gpt_neo import FMLoRA_GPT_Util, ElasticDNN_OfflineTextGenFMModel, ElasticDNN_OfflineTextGenMDModel, FM_to_MD_GPT_Util, collate_fn
# from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
# from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
# from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation, LayerActivation2, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_gen_scenario
from utils.common.log import logger
from utils.dl.common.loss import CrossEntropyLossSoft2
import torch.nn.functional as F
import sys

class ElasticDNN_GPT_OfflineTextGenFMModel(ElasticDNN_OfflineTextGenFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        return FM_to_MD_GPT_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples)

    def generate_md_by_reducing_layers(self, layers, samples: torch.Tensor):
        return FM_to_MD_GPT_Util().init_md_from_fm_by_reducing_layers_with_perf_test(self.models_dict['main'], 
                                                                        layers, samples)

    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'model.lm_head'), True, self.device)
    
    def get_feature_hooks(self, layers=None):
        res = {}
        res['head'] = LayerActivation(get_module(self.models_dict['main'], 'model.lm_head'), True, self.device)
        res['hiddens'] = []
        for block_i, _ in enumerate(self.models_dict['main'].model.transformer.h):
            if layers is None or block_i in layers:       
                res['hiddens'].append(LayerActivation(get_module(self.models_dict['main'], f'model.transformer.h.{block_i}'), True, self.device))
        return res

    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        raise NotImplementedError
    
    def forward_to_get_task_loss(self, x, y):
        self.to_train_mode()
        return self.infer(x)
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_GPT_Util()

    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'model.lm_head')
        params_name = {k for k, v in head.named_parameters()}
        logger.info(f'task head params: {params_name}')
        return list(head.parameters())
        
        
class ElasticDNN_GPT_OfflineTextGenMDModel(ElasticDNN_OfflineTextGenMDModel):
    def __init__(self, name: str, models_dict_path: str, device: str):
        super().__init__(name, models_dict_path, device)
        
        self.distill_criterion = CrossEntropyLossSoft2()
        # self.distill_criterion = nn.KLDivLoss(reduction="batchmean")
        self.hidden_criterion = nn.MSELoss()
        
    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'model.lm_head'))
    
    def get_feature_hooks(self):
        res = {}
        res['head'] = LayerActivation2(get_module(self.models_dict['main'], 'model.lm_head'))
        res['hiddens'] = [LayerActivation(get_module(self.models_dict['main'], f'model.transformer.h.{block_i}'), True, self.device) for block_i, _ in enumerate(self.models_dict['main'].model.transformer.h)]
        return res

    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        return self.infer(x)
    
    def get_distill_loss(self, student_hook, teacher_hook, loss_mask):
        student_output = student_hook['head'].output
        teacher_output = teacher_hook['head'].output

        t = 1
        # print(student_output, teacher_output)
        loss_logits = self.distill_criterion(
            student_output / t,  # vocab_size
            teacher_output / t,
            loss_mask
        ) * (t ** 2)

        loss_hid = 0.
        # num_layers = 0
        # for stu_hid, tea_hid in zip(student_hook['hiddens'], teacher_hook['hiddens']):
        #     loss_hid += self.hidden_criterion(stu_hid.output, tea_hid.output)
        #     num_layers += 1
        # loss_hid /= num_layers

        loss = loss_logits + loss_hid

        return loss
    
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module): # TODO:
        if any([k in self_param_name for k in ['fbs', 'wte', 'wpe']]):
            return None
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if 'q_proj' in self_param_name or 'k_proj' in self_param_name or 'v_proj' in self_param_name:
            ss = self_param_name.split('.')
            raise NotImplementedError() # TODO:
            fm_qkv_name = '.'.join(ss[0: -2]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -2]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0) # task-specific params (LoRA)
            ], dim=0)
            
        elif 'to_qkv.bias' in self_param_name:
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
            return get_parameter(fm, fm_qkv_name)
            
        elif 'mlp.c_fc' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)

        else:
            return get_parameter(fm, self_param_name)
        
        
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    torch.cuda.set_device(1)
    # 3. init scenario
    os.environ['gpt_neo_series_id'] = gpt_neo_series_id
    
    scenario = build_gen_scenario(
        source_datasets_name=['No_robots'],
        target_datasets_order=['No_robots'] * 1, # TODO
        da_mode='close_set',
        data_dirs={
            'No_robots': f'/data/zql/datasets/no_robots',
        },
    )
    
    # 1. init model
    
    fm_models_dict_path = 'new_impl/nlp/gpt-neo/text_generation/results/gen_lora.py/20231210/999999-205643-results/models/fm_last.pt'
    fm_models = torch.load(fm_models_dict_path)
    fm_models_dict_path = save_models_dict_for_init(fm_models, __file__, 'fm_gpt_txt_gen_lora')
    md_models_dict_path = save_models_dict_for_init({
        'main': -1
    }, __file__, 'md_gpt_none')
    device = 'cuda'
    
    fm_model = ElasticDNN_GPT_OfflineTextGenFMModel('fm', fm_models_dict_path, device)
    md_model = ElasticDNN_GPT_OfflineTextGenMDModel('md', md_models_dict_path, device)
    
    # 2. init alg
    models = {
        'fm': fm_model,
        'md': md_model
    }
    fm_to_md_alg = ElasticDNN_MDPretrainingWoFBSAlg(models, get_res_save_dir(__file__, None))
    
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_to_md_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': {'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]).to(device), 
                                  'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device), 
                                  'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device), 'return_dict': False},
        # 'generate_md_layers': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23],
        'generate_md_width_ratio': 8,
        'train_batch_size': 4,
        'val_batch_size': 1,
        'num_workers': 4,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 320000)},
        'num_iters': 320000,
        'val_freq': 1000,
        'distill_loss_weight': 0.6
    }, collate_fn=collate_fn)
    