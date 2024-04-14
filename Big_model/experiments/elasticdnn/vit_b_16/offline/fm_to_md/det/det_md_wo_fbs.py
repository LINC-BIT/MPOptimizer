import torch
import sys
from torch import nn
from methods.elasticdnn.api.model import ElasticDNN_OfflineDetFMModel, ElasticDNN_OfflineDetMDModel
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.vit import FMLoRA_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation2, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F
from utils.common.log import logger


class ElasticDNN_ViT_OfflineDetFMModel(ElasticDNN_OfflineDetFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        return FM_to_MD_ViT_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples)
        
    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'blocks.11'))
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticViTUtil()
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        # return F.cross_entropy(self.infer(x), y)
        self.to_train_mode()
        return self.models_dict['main'](x, y)['total_loss']
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_ViT_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'head')
        params_name = {k for k, v in head.named_parameters()}
        logger.info(f'task head params: {params_name}')
        return list(head.parameters())
        
        
class ElasticDNN_ViT_OfflineDetMDModel(ElasticDNN_OfflineDetMDModel):
    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'blocks.11'))
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        return self.models_dict['main'](x, y)['total_loss']
    
    def get_distill_loss(self, student_output, teacher_output):
        # print(student_output, teacher_output)
        return F.mse_loss(student_output, teacher_output)
    
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module):
        if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
            return None
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if 'to_qkv.weight' in self_param_name:
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -2]) + '.abs'
            fm_abs = get_module(fm, fm_abs_name)
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0) # task-specific params (LoRA)
            ], dim=0)
            
        elif 'to_qkv.bias' in self_param_name:
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
            return get_parameter(fm, fm_qkv_name)
            
        elif 'mlp.fc1' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)

        else:
            return get_parameter(fm, self_param_name)
        
        
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    # 3. init scenario
    scenario = build_scenario(
        source_datasets_name=['GTA5Det', 'SuperviselyPersonDet'],
        target_datasets_order=['CityscapesDet', 'BaiduPersonDet'] * 10,
        da_mode='close_set',
        data_dirs={
            'GTA5Det': '/data/zql/datasets/GTA-ls-copy/GTA5',
            'SuperviselyPersonDet': '/data/zql/datasets/supervisely_person_full_20230635/Supervisely Person Dataset',
            'CityscapesDet': '/data/zql/datasets/cityscape/',
            'BaiduPersonDet': '/data/zql/datasets/baidu_person/clean_images/'
        },
    )
    
    # 1. init model
    # from dnns.deeplabv3.head import modify_forward_head
    # modify_forward_head() # TODO: bring a bug
    from dnns.vit import vit_b_16
    fm_models_dict_path = 'experiments/elasticdnn/vit_b_16/offline/fm_lora/det/results/det.py/20230627/star_final_999983-140249-3_class_det_trial_lr1e-4_use_cls_lora_ckpt/models/fm_best.pt'
    fm_models = torch.load(fm_models_dict_path)
    fm_models_dict_path = save_models_dict_for_init(fm_models, __file__, 'fm_vit_b_16_det_lora')
    md_models_dict_path = save_models_dict_for_init({
        'main': -1
    }, __file__, 'md_vit_b_16_none')
    device = 'cuda'
    
    fm_model = ElasticDNN_ViT_OfflineDetFMModel('fm', fm_models_dict_path, device, scenario.num_classes)
    md_model = ElasticDNN_ViT_OfflineDetMDModel('md', md_models_dict_path, device, scenario.num_classes)
    
    # 2. init alg
    models = {
        'fm': fm_model,
        'md': md_model
    }
    fm_to_md_alg = ElasticDNN_MDPretrainingWoFBSAlg(models, get_res_save_dir(__file__, None))
    
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_to_md_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': (1, 3, 224, 224),
        'generate_md_width_ratio': 4,
        
        'train_batch_size': 32,
        'val_batch_size': 32,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        'num_iters': 80000,
        'val_freq': 4000,
        'distill_loss_weight': 1e-4
    })
    