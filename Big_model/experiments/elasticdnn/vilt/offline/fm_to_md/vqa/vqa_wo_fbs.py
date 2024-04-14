import torch
from methods.elasticdnn.api.model import ElasticDNN_OfflineVQAFMModel, ElasticDNN_OfflineVQAMDModel
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.vilt import FMLoRA_Vilt_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vilt import FM_to_MD_Vilt_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.common.log import logger
import torch.nn.functional as F
import sys
from torch import nn
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg



class ElasticDNN_Vilt_OfflineVQAFMModel(ElasticDNN_OfflineVQAFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        return FM_to_MD_Vilt_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples)
        # raise NotImplementedError

    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier.3'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        raise NotImplementedError
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        # print(x['input_ids'].size(), x['pixel_values'].size(), )
        o = self.infer(x).logits
        
        # print(o.size(), y.size(), o, y)
        
        return F.binary_cross_entropy_with_logits(o, y) * y.shape[1]
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_Vilt_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'classifier')
        params_name = {k for k, v in head.named_parameters()}
        logger.info(f'task head params: {params_name}')
        return list(head.parameters())
        
        
class ElasticDNN_Vilt_OfflineVQAMDModel(ElasticDNN_OfflineVQAMDModel):
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier.3'), True, self.device)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        # print(x['input_ids'].size(), x['pixel_values'].size(), )
        o = self.infer(x).logits
        
        # print(o.size(), y.size(), o, y)
        
        return F.binary_cross_entropy_with_logits(o, y) * y.shape[1]
    
    def get_distill_loss(self, student_output, teacher_output):
        # print(student_output, teacher_output)
        return F.mse_loss(student_output, teacher_output.detach())
    
    def get_trained_params(self):
        return self.models_dict['main'].parameters()
    
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module): # TODO:
        if any([k in self_param_name for k in ['fbs', 'ab', 'embeddings']]):
            return None
        
        p = get_parameter(self.models_dict['main'], self_param_name)
        if p.dim() == 0:
            return None
        
        elif p.dim() == 1 and 'LayerNorm' in self_param_name and 'weight' in self_param_name:
            # if self_param_name.startswith('norm'):
            #     return None
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

        else:
            return get_parameter(fm, self_param_name)
    
    
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    scenario = build_scenario(
        source_datasets_name=['VQAv2_split1'],
        target_datasets_order=['VQAv2_split1_c'] * 1, # TODO
        da_mode='close_set',
        data_dirs={
            'VQAv2_split1': '/data/zql/datasets/vqav2',
            'VQAv2_split1_c': '/data/zql/datasets/vqav2'
        },
    )
    
    # 1. init model
    fm_models_dict_path = 'experiments/elasticdnn/vilt/offline/fm_lora/vqa/results/vqa.py/20230730/999971-190508-trial/models/fm_best.pt'
    fm_models = torch.load(fm_models_dict_path)
    fm_models_dict_path = save_models_dict_for_init(fm_models, __file__, 'fm_vilt_vqa_lora')
    md_models_dict_path = save_models_dict_for_init({
        'main': -1
    }, __file__, 'md_vilt_none')
    device = 'cuda'
    
    fm_model = ElasticDNN_Vilt_OfflineVQAFMModel('fm', fm_models_dict_path, device)
    md_model = ElasticDNN_Vilt_OfflineVQAMDModel('md', md_models_dict_path, device)
    
    # 2. init alg
    models = {
        'fm': fm_model,
        'md': md_model
    }
    import sys
    
    # from experiments.elasticdnn.clip.offline.fm_to_md.cls.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
    # from methods.elasticdnn.api.algs.md_pretraining_wo_fbs_clip_debug import ElasticDNN_MDPretrainingWoFBSAlg
    fm_to_md_alg = ElasticDNN_MDPretrainingWoFBSAlg(models, get_res_save_dir(__file__, sys.argv[1]))
    
    # sample_dataset = list(scenario.get_offline_datasets().values())[0]['train']
    
    sample_dataset = list(scenario.get_offline_datasets().values())[0]['train']
    sample = sample_dataset[0][0]
    
    for k, v in sample.items():
        sample[k] = v.unsqueeze(0)
    
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_to_md_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': sample,
        'generate_md_width_ratio': 4,
        
        'train_batch_size': 64,
        'val_batch_size': 256,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        'num_iters': 80000,
        'val_freq': 1000,
        'distill_loss_weight': 1.
    })
    