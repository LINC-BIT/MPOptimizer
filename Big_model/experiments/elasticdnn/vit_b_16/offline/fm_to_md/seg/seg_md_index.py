import torch
import sys
from torch import nn
from methods.elasticdnn.api.model import ElasticDNN_OfflineSegFMModel, ElasticDNN_OfflineSegMDModel
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.vit import FMLoRA_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F


class ElasticDNN_ViT_OfflineSegFMModel(ElasticDNN_OfflineSegFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        return FM_to_MD_ViT_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples).to(self.device)
        
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'head'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticViTUtil()
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(x), y)
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_ViT_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'norm.1.head')
        return list(head.parameters())
        
        
class ElasticDNN_ViT_OfflineSegMDModel(ElasticDNN_OfflineSegMDModel):
        
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'head'), True, self.device)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(x), y)
    
    def get_distill_loss(self, student_output, teacher_output):
        return F.mse_loss(student_output, teacher_output)
    
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module):
        # only between qkv.weight, norm.weight/bias
        if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
            return None
        
        p = get_parameter(self.models_dict['main'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1 and 'norm' in self_param_name and 'weight' in self_param_name:
            if self_param_name.startswith('norm'):
                self_param_name = self_param_name.replace('norm', 'norm.0')
            return get_parameter(fm, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if 'qkv.weight' in self_param_name:
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.qkv'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.abs'
            fm_abs = get_module(fm, fm_abs_name)
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0) # task-specific params (LoRA)
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
            return get_parameter(fm, fm_param_name)
        
        else:
            # return get_parameter(fm, self_param_name)
            return None
        
        
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    # 3. init scenario
    scenario = build_scenario(
        source_datasets_name=['GTA5', 'SuperviselyPerson'],
        target_datasets_order=['Cityscapes', 'BaiduPerson'] * 10,
        da_mode='close_set',
        data_dirs={
            'GTA5': '/data/zql/datasets/GTA-ls-copy/GTA5',
            'SuperviselyPerson': '/data/zql/datasets/supervisely_person/Supervisely Person Dataset',
            'Cityscapes': '/data/zql/datasets/cityscape/',
            'BaiduPerson': '/data/zql/datasets/baidu_person/clean_images/'
        },
    )
    
    # 1. init model
    # from dnns.deeplabv3.head import modify_forward_head
    # modify_forward_head() # TODO: bring a bug
    from dnns.vit import vit_b_16
    fm_models_dict_path = 'experiments/elasticdnn/vit_b_16/offline/fm_to_md/seg/results/seg_md_w_fbs.py/20230611/999999-183508-new_md_structure/models/fm_best.pt'
    fm_models = torch.load(fm_models_dict_path)
    from utils.dl.common.model import set_module
    set_module(
        fm_models['main'],
        'norm',
        nn.Sequential(
            get_module(fm_models['main'], 'norm'),
            get_module(fm_models['main'], 'head')
        )
    )
    set_module(fm_models['main'], 'head', nn.Identity())
    fm_models['main'].forward = fm_models['main'].forward_features
    fm_models_dict_path = save_models_dict_for_init(fm_models, __file__, 'fm_vit_b_16_seg_lora')

    pretrained_md_models_dict_path = 'experiments/elasticdnn/vit_b_16/offline/fm_to_md/seg/results/seg_md_wo_fbs.py/20230611/999998-141814/models/md_best.pt'
    md_models_dict_path = save_models_dict_for_init(torch.load(pretrained_md_models_dict_path), __file__, 'md_vit_b_16_seg_pretrained_w_fbs')
    device = 'cuda'
    
    fm_model = ElasticDNN_ViT_OfflineSegFMModel('fm', fm_models_dict_path, device, scenario.num_classes)
    md_model = ElasticDNN_ViT_OfflineSegMDModel('md', md_models_dict_path, device, scenario.num_classes)
    
    # 2. init alg
    models = {
        'fm': fm_model,
        'md': md_model
    }
    from methods.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md import ElasticDNN_MDPretrainingIndexAlg
    fm_to_md_alg = ElasticDNN_MDPretrainingIndexAlg(models, get_res_save_dir(__file__, sys.argv[1]))
    
    
    
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_to_md_alg.run(scenario, hyps={
        # 'launch_tbboard': False,
        
        # 'samples_size': (1, 3, 224, 224),
        # 'generate_md_width_ratio': 2,
        
        # 'train_batch_size': 16,
        # 'val_batch_size': 128,
        # 'num_workers': 16,
        # 'optimizer': 'AdamW',
        # 'optimizer_args': {'lr': 5e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        # 'scheduler': 'LambdaLR',
        # 'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        # 'num_iters': 80000,
        # 'val_freq': 4000,
        # 'distill_loss_weight': 1.0
        
        'launch_tbboard': False,
        
        'samples_size': (1, 3, 224, 224),
        
        'FBS_r': 16,
        'FBS_ignore_layers': [],
        
        'train_batch_size': 16,
        'val_batch_size': 128,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-3 * 16 / 256, 'betas': [0.9, 0.999], 'weight_decay': 5e-4},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        'indexes_optimizer_args': {'lr': 3e-3, 'momentum': 0.9, 'weight_decay': 5e-4},
        # 'scheduler': 'StepLR',
        # 'scheduler_args': {'step_size': 40000, 'gamma': 0.1},
        'num_iters': 60000,
        'val_freq': 500,
        'max_sparsity': 0.9,
        'min_sparsity': 0.0,
        'l1_reg_loss_weight': 1e-9,
        'index_loss_weight': 1e-4,
        'val_num_sparsities': 4,
        
        'bn_cal_num_iters': 0,
        
        'index_init': 'zero',
        'index_guided_linear_comb_split_size': 512
    })
    