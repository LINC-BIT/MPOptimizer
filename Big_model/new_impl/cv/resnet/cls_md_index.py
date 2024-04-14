import torch
import sys
from torch import nn
from dnns.vit import make_softmax_prunable
from new_impl.cv.elasticdnn.api.model import ElasticDNN_OfflineClsFMModel, ElasticDNN_OfflineClsMDModel
# from methods.elasticdnn.api.algs.md_pretraining_w_fbs import ElasticDNN_MDPretrainingWFBSAlg
from new_impl.cv.elasticdnn.model.base import ElasticDNNUtil
from new_impl.cv.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
# from beit import FM_to_MD_beit_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
#from beit import FMLoRA_beit_Util
from resnet import ElasticresUtil
from utils.dl.common.model import LayerActivation, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F


class ElasticDNN_res_OfflineClsFMModel(ElasticDNN_OfflineClsFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        # return FM_to_MD_beit_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
        #                                                                 reducing_width_ratio, samples).to(self.device)
        raise NotImplementedError
        
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticresUtil()
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(x).logits, y)
    
    def get_lora_util(self) -> FMLoRA_Util:
        #return FMLoRA_beit_Util()
        raise NotImplementedError
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'classifier')
        return list(head.parameters())

        
class ElasticDNN_res_OfflineClsMDModel(ElasticDNN_OfflineClsMDModel):
    def __init__(self, name: str, models_dict_path: str, device: str):
        super().__init__(name, models_dict_path, device)
        
        self.distill_criterion = CrossEntropyLossSoft()
        
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier'), True, self.device)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(x).logits, y)
    
    def get_distill_loss(self, student_output, teacher_output):
        return self.distill_criterion(student_output, teacher_output)
    
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module):
        # only between qkv.weight, norm.weight/bias
        if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
            return None
        
        p = get_parameter(self.models_dict['main'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1 and 'layernorm' in self_param_name and 'weight' in self_param_name:
            return get_parameter(fm, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        # if 'qkv.weight' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv'
        #     fm_qkv = get_module(fm, fm_qkv_name)
            
        #     fm_abs_name = '.'.join(ss[0: -2]) + '.abs'
        #     fm_abs = get_module(fm, fm_abs_name)
            
        #     return torch.cat([
        #         fm_qkv.weight.data, # task-agnositc params
        #         torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0) # task-specific params (LoRA)
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
        
        # if 'qkv.weight' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -1]) + '.qkv'
        #     fm_qkv = get_module(fm, fm_qkv_name)
            
        #     fm_abs_name = '.'.join(ss[0: -1]) + '.abs'
        #     fm_abs = get_module(fm, fm_abs_name)
            
        #     return torch.cat([
        #         fm_qkv.weight.data, # task-agnositc params
        #         torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0) # task-specific params (LoRA)
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
        #     res = get_parameter(fm, fm_param_name)
        #     # print('mlp fc2 debug', fm_param_name, res is None)
        #     return res
        
        # else:
        #     # return get_parameter(fm, self_param_name)
        #     return None
    
    # def get_matched_param_of_fm(self, self_param_name, fm: nn.Module):
    #     if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
    #         return None
        
    #     # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
    #     if 'to_qkv.weight' in self_param_name:
    #         ss = self_param_name.split('.')
            
    #         fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv'
    #         fm_qkv = get_module(fm, fm_qkv_name)
            
    #         fm_abs_name = '.'.join(ss[0: -2]) + '.abs'
    #         fm_abs = get_module(fm, fm_abs_name)
            
    #         return torch.cat([
    #             fm_qkv.weight.data, # task-agnositc params
    #             torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0) # task-specific params (LoRA)
    #         ], dim=0)
            
    #     elif 'to_qkv.bias' in self_param_name:
    #         ss = self_param_name.split('.')
            
    #         fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
    #         return get_parameter(fm, fm_qkv_name)
            
    #     elif 'mlp.fc1' in self_param_name:
    #         fm_param_name = self_param_name.replace('.linear', '')
    #         return get_parameter(fm, fm_param_name)

    #     else:
    #         return get_parameter(fm, self_param_name)
        if ('attention.attention.query' in self_param_name or 'attention.attention.key' in self_param_name or \
            'attention.attention.value' in self_param_name) and ('weight' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs[1].weight @ fm_abs[0].weight
            ], dim=0)
            
        elif ('attention.attention.query' in self_param_name or 'attention.attention.key' in self_param_name or \
            'attention.attention.value' in self_param_name) and ('bias' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc.bias'
            return get_parameter(fm, fm_qkv_name)
            
        elif 'intermediate.dense' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)
        elif 'output.dense' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)
        else:
            #return get_parameter(fm, self_param_name)
            return None
               
        
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    # 1. init model
    #from dnns.vit import vit_b_16
    fm_models_dict_path = 'new_impl/cv/resnet/results/cls.py/20231103/999999-100700-/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/resnet/cls.py/models/fm_best.pt'
    fm_models_dict_path = save_models_dict_for_init(torch.load(fm_models_dict_path), __file__, 'fm_beit_cls_lora')
    pretrained_md_models_dict_path = 'new_impl/cv/resnet/results/cls.py/20231103/999999-100700-/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/resnet/cls.py/models/fm_best.pt'
    md_models_dict = torch.load(pretrained_md_models_dict_path)
    md_models_dict_path = save_models_dict_for_init(md_models_dict, __file__, 'md_res_cls_pretrained_wo_fbs')
    torch.cuda.set_device(1)
    device = 'cuda'
    
    fm_model = ElasticDNN_res_OfflineClsFMModel('fm', fm_models_dict_path, device)
    md_model = ElasticDNN_res_OfflineClsMDModel('md', md_models_dict_path, device)
    
    # 2. init alg
    models = {
        'fm': fm_model,
        'md': md_model
    }
    from new_impl.cv.elasticdnn.api.algs.md_pretraining_index_cnn import ElasticDNN_MDPretrainingIndexAlg

    fm_to_md_alg = ElasticDNN_MDPretrainingIndexAlg(models, get_res_save_dir(__file__, sys.argv[0]))
    
    # 3. init scenario
    scenario = build_scenario(
        source_datasets_name=['GTA5Cls', 'SuperviselyPersonCls'],
        target_datasets_order=['CityscapesCls', 'BaiduPersonCls'] * 15,
        da_mode='close_set',
        data_dirs={
            'GTA5Cls': '/data/zql/datasets/gta5_for_cls_task',
            'SuperviselyPersonCls': '/data/zql/datasets/supervisely_person_for_cls_task',
            'CityscapesCls': '/data/zql/datasets/cityscapes_for_cls_task',
            'BaiduPersonCls': '/data/zql/datasets/baidu_person_for_cls_task'
        },
    )
    
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_to_md_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': (1, 3, 224, 224),
        
        'FBS_r': 16,
        'FBS_ignore_layers': [],
        
        'train_batch_size': 128,
        'val_batch_size': 512,
        'num_workers': 16,
        'optimizer': 'AdamW',
        # 'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'indexes_optimizer_args': {'lr': 3e-3, 'betas': [0.9, 0.999], 'weight_decay': 0.1},
        # 'scheduler': 'StepLR',
        # 'scheduler_args': {'step_size': 20000, 'gamma': 0.1},
        # 'optimizer': 'AdamW',
        # 'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'optimizer_args': {'lr': 1e-3, 'betas': [0.9, 0.999], 'weight_decay': 0.01},#注意学习率的调整，不同的模型不一样。
        
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        
        'max_sparsity': 0.9,
        'min_sparsity': 0.0,
        'num_iters': 60000,
        'val_freq': 900,
        'index_loss_weight': 1e-4,
        'l1_reg_loss_weight': 1e-9,
        'val_num_sparsities': 4,
        
        'bn_cal_num_iters': 800,#有bn层注意需要加上这个
        
        'index_init': 'zero',
        'index_guided_linear_comb_split_size': 512
    })
    