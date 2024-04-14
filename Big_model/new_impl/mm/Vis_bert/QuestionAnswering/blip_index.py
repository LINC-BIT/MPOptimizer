import torch
import sys
from torch import nn
from new_impl.cv.elasticdnn.api.model import ElasticDNN_OfflineVQAFMModel, ElasticDNN_OfflineVQAMDModel
from new_impl.cv.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from new_impl.cv.elasticdnn.model.base import ElasticDNNUtil
from new_impl.cv.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from blip import FM_to_MD_blip_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from blip import FMLoRA_blip_Util
from blip import ElasticblipUtil
from utils.dl.common.model import LayerActivation2, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F
from utils.common.log import logger


class ElasticDNN_blip_OfflineVQAFMModel(ElasticDNN_OfflineVQAFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor): # TODO:
        return FM_to_MD_blip_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples)
        # raise NotImplementedError

    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'text_decoder.cls.predictions.decoder'))
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil: # TODO:
        return ElasticblipUtil()
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        # print(x['input_ids'].size(), x['pixel_values'].size(), )
        #o = self.infer(x)
        o = self.models_dict['main'](**y)
        # print(o.size(), y.size(), o, y)
        
        #return F.binary_cross_entropy_with_logits(o, y) * y.shape[1]
        return o.loss
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_blip_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'text_decoder.cls.predictions.decoder')
        params_name = {k for k, v in head.named_parameters()}
        logger.info(f'task head params: {params_name}')
        return list(head.parameters())
        
        
class ElasticDNN_blip_OfflineVQAMDModel(ElasticDNN_OfflineVQAMDModel):
    # def __init__(self, name: str, models_dict_path: str, device: str):
    #     super().__init__(name, models_dict_path, device)
        
    #     self.distill_criterion = CrossEntropyLossSoft()
        
    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'text_decoder.cls.predictions.decoder'))
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        # print(x['input_ids'].size(), x['pixel_values'].size(), )
        #o = self.infer(x)
        o = self.models_dict['main'](**y)
        # print(o.size(), y.size(), o, y)
        
        #return F.binary_cross_entropy_with_logits(o, y) * y.shape[1]
        return o.loss
    
    def get_distill_loss(self, student_output, teacher_output):
        # print(student_output, teacher_output)
        return F.mse_loss(student_output, teacher_output.detach())
    
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module): # TODO:
        if any([k in self_param_name for k in ['fbs', 'ab', 'embeddings']]):
            return None
        
        p = get_parameter(self.models_dict['main'], self_param_name)
        if p.dim() == 0:
            return None
        
        elif p.dim() == 1 and 'LayerNorm' in self_param_name.lower() and 'weight' in self_param_name:
            # if self_param_name.startswith('norm'):
            #     return None
            return get_parameter(fm, self_param_name)
        elif p.dim() == 1 and 'norm' in self_param_name and 'weight' in self_param_name:
            return get_parameter(fm, self_param_name)
        
        # 1. xx.query.weight -> xx.query.fc.weight and xx.query.ab.0/1
        if ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('weight' in self_param_name):
            ss = self_param_name.split('.')
            # raise NotImplementedError() # TODO:
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs[1].weight @ fm_abs[0].weight
            ], dim=0)
            
        elif ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('bias' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc.bias'
            return get_parameter(fm, fm_qkv_name)
            
        elif 'intermediate.dense' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)
        
        elif 'qkv.weight' in self_param_name:
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs[1].weight @ fm_abs[0].weight # task-specific params (LoRA)
            ], dim=0)
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
            
        elif 'mlp.fc1' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)

        elif 'mlp.fc2' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name
            res = get_parameter(fm, fm_param_name)
            # print('mlp fc2 debug', fm_param_name, res is None)
            return res
        else:
            #return get_parameter(fm, self_param_name)
            return None
        
        
        
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    # 3. init scenario
    scenario = build_scenario(
        source_datasets_name=['VQA_split1'],
        target_datasets_order=['VQA_split1_c'] * 1, # TODO
        da_mode='close_set',
        data_dirs={
            'VQA_split1': '/data/zql/datasets/vqav2',
            'VQA_split1_c': '/data/zql/datasets/vqav2'
        },
    )
    
    # 1. init model
    # from dnns.deeplabv3.head import modify_forward_head
    # modify_forward_head() # TODO: bring a bug
    # from dnns.vit import vit_b_16
    fm_models_dict_path = 'new_impl/mm/Vis_bert/QuestionAnswering/results/blip_fbs.py/20231020/999999-162038-/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/mm/Vis_bert/QuestionAnswering/blip_fbs.py/models/fm_best.pt'
    fm_models = torch.load(fm_models_dict_path)
    fm_models_dict_path = save_models_dict_for_init(fm_models, __file__, 'fm_blip_vqa_lora')
    md_models_dict_path = save_models_dict_for_init(
        torch.load('new_impl/mm/Vis_bert/QuestionAnswering/results/blip_fbs.py/20231020/999999-162038-/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/mm/Vis_bert/QuestionAnswering/blip_fbs.py/models/md_best.pt'), 
        __file__, 'md_blip_vqa_raw_pretrained')
    device = 'cuda'
    
    fm_model = ElasticDNN_blip_OfflineVQAFMModel('fm', fm_models_dict_path, device)
    md_model = ElasticDNN_blip_OfflineVQAMDModel('md', md_models_dict_path, device)
    
    # 2. init alg
    models = {
        'fm': fm_model,
        'md': md_model
    }
    from new_impl.cv.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md import ElasticDNN_MDPretrainingIndexAlg
    fm_to_md_alg = ElasticDNN_MDPretrainingIndexAlg(models, get_res_save_dir(__file__, sys.argv[0]))

    sample_dataset = list(scenario.get_offline_datasets().values())[0]['train']
    sample = sample_dataset[0][0]
    
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_to_md_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': sample,
        
        'FBS_r': 8,
        'FBS_ignore_layers': [],
        
        'train_batch_size': 16,
        'val_batch_size': 256,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        
        'indexes_optimizer_args': {'lr': 3e-3, 'momentum': 0.9, 'weight_decay': 5e-4},
        
        'num_iters': 80000,
        'val_freq': 20,
        
        'max_sparsity': 0.9,
        'min_sparsity': 0.0,
        'l1_reg_loss_weight': 1e-9,
        'index_loss_weight': 1e-4,
        'val_num_sparsities': 4,
        
        'bn_cal_num_iters': 0,
        
        'index_init': 'zero',
        'index_guided_linear_comb_split_size': 512
    })
    