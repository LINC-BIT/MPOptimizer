import torch
import torch.nn as nn
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from glip import ElasticGLIPUtil, FMLoRA_GLIP_Util, FM_to_MD_GLIP_Util, ElasticDNN_OfflineMMDetFMModel, ElasticDNN_OfflineMMDetMDModel
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from new_impl.cv.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md_glip import ElasticDNN_MDPretrainingIndexAlg
from utils.dl.common.model import LayerActivation3, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
import torch.nn.functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList
import os
from utils.dl.common.loss import CrossEntropyLossSoft

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class ElasticDNN_GLIP_OfflineMMDetFMModel(ElasticDNN_OfflineMMDetFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        return FM_to_MD_GLIP_Util().init_md_from_fm_by_reducing_width_with_perf_test_2(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples).to(self.device)
        
    def get_feature_hook(self) -> LayerActivation3:
        return LayerActivation3(get_module(self.models_dict['main'], 'model.rpn'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticGLIPUtil()
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        loss_dict = self.infer(x)
        losses = sum(loss for loss in loss_dict.values())
        # print(losses)
        
        return losses
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_GLIP_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'classifier')
        return list(head.parameters())

        
class ElasticDNN_GLIP_OfflineMMDetMDModel(ElasticDNN_OfflineMMDetMDModel):
    def __init__(self, name: str, models_dict_path: str, device: str, collate_fn=None):
        super().__init__(name, models_dict_path, device, collate_fn=collate_fn)
        
        self.distill_criterion = CrossEntropyLossSoft()
        
    def get_feature_hook(self) -> LayerActivation3:
        return LayerActivation3(get_module(self.models_dict['main'], 'model.rpn'), True, self.device)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        loss_dict = self.infer(x)
        losses = sum(loss for loss in loss_dict.values())
        # print(losses)
        
        return losses
    
    def get_distill_loss(self, student_output, teacher_output):
        
        return self.distill_criterion(student_output, teacher_output)
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module):
        # only between qkv.weight, norm.weight/bias
        if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed', 'conv']]):
            return None
        
        p = get_parameter(self.models_dict['main'], self_param_name)

        if p.dim() == 0:
            return None
        # elif p.dim() == 1 and 'norm' in self_param_name and 'weight' in self_param_name:
        #     return get_parameter(fm, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
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
        
        elif ('attn.qkv' in self_param_name or \
        'attn.v_proj' in self_param_name or 'attn.l_proj' in self_param_name or 'attn.values_v_proj' in self_param_name or 'attn.values_l_proj' in self_param_name) and ('weight' in self_param_name):        
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = [get_module(fm, fm_abs_name)]
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0) # task-specific params (LoRA)
            ], dim=0)
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
        elif ('attn.qkv' in self_param_name or \
        'attn.v_proj' in self_param_name or 'attn.l_proj' in self_param_name or 'attn.values_v_proj' in self_param_name or 'attn.values_l_proj' in self_param_name) and ('bias' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc.bias'
            return get_parameter(fm, fm_qkv_name)
                    
        elif ('intermediate.dense' in self_param_name or 'mlp.fc1' in self_param_name) and ('weight' in self_param_name or 'bias' in self_param_name):
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)
        
        else:
            # return get_parameter(fm, self_param_name)
            return None        
                
        
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    # 1. init model
    from dnns.vit import vit_b_16
    fm_models_dict_path = 'new_impl/cv/glip/object_detection/results/det_md_wo_fbs.py/20231129/999999-153230-results/models/fm_best.pt'
    fm_models_dict_path = save_models_dict_for_init(torch.load(fm_models_dict_path), __file__, 'fm_glip_cls_lora')
    pretrained_md_models_dict_path = 'new_impl/cv/glip/object_detection/results/det_md_wo_fbs.py/20231129/999999-153230-results/models/md_best.pt'
    md_models_dict = torch.load(pretrained_md_models_dict_path)
    md_models_dict_path = save_models_dict_for_init(md_models_dict, __file__, 'md_glip_cls_pretrained_wo_fbs')
    torch.cuda.set_device(1)
    device = 'cuda'
    
    from glip import glip_model, build_transform, run_ner, collect_mm_fn

    fm_model = ElasticDNN_GLIP_OfflineMMDetFMModel('fm', fm_models_dict_path, device, collate_fn=collect_mm_fn)
    md_model = ElasticDNN_GLIP_OfflineMMDetMDModel('md', md_models_dict_path, device, collate_fn=collect_mm_fn)
    
    # 2. init alg
    models = {
        'fm': fm_model,
        'md': md_model
    }

    fm_to_md_alg = ElasticDNN_MDPretrainingIndexAlg(models, get_res_save_dir(__file__, "results"))
    
    # 3. init scenario
    scenario = build_scenario(
        source_datasets_name=['MM-COCO2017'],
        target_datasets_order=['MM-CityscapesDet'],
        da_mode='close_set',
        data_dirs={
            'MM-COCO2017': '/data/zql/datasets/coco2017',
            'MM-CityscapesDet': '/data/zql/datasets/cityscape'
        },
    )
    
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    from PIL import Image, ImageDraw
    import requests
    import numpy as np
    from evaluator import MMCOCODecoder
    cfg_path = 'new_impl/cv/glip/object_detection/pretrained_model/glip_Swin_T_O365_GoldG.yaml'
    model_path = 'new_impl/cv/glip/object_detection/pretrained_model/glip_tiny_model_o365_goldg_cc_sbu.pth'
    config, _ = glip_model(cfg_path, model_path)
    transform = build_transform(config, None)
    ori_image = Image.open('new_impl/cv/glip/object_detection/000000103759.jpg').convert("RGB")
    image = transform(np.asarray(ori_image)[:, :, [2, 1, 0]])
    text = 'orange. umbrella. '
    targets = BoxList(torch.FloatTensor([[0., 0., 0., 0.]]), image_size=image.size()[1:], mode='xyxy')
    targets.add_field('caption', text)
    targets.add_field('tokens_positive', run_ner(text))
    targets.add_field('labels', torch.LongTensor([0]))
    samples = {'images' : [image], 'targets' : [targets]}

    fm_to_md_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': samples,
        
        'FBS_r': 16,
        'FBS_ignore_layers': [],
        
        'train_batch_size': 16,
        'val_batch_size': 1,
        'num_workers': 16,
        'optimizer': 'AdamW',
        # 'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'indexes_optimizer_args': {'lr': 3e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.1},
        # 'scheduler': 'StepLR',
        # 'scheduler_args': {'step_size': 20000, 'gamma': 0.1},
        # 'optimizer': 'AdamW',
        # 'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'optimizer_args': {'lr': 2e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},#注意学习率的调整，不同的模型不一样。
        'transform' : transform,
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        
        'max_sparsity': 0.9,
        'min_sparsity': 0.3,
        'num_iters': 9000,
        'val_freq': 100,
        'index_loss_weight': 1e-4,
        'l1_reg_loss_weight': 1e-9,
        'val_num_sparsities': 4,
        
        'bn_cal_num_iters': 0,#有bn层注意需要加上这个
        
        'index_init': 'zero',
        'index_guided_linear_comb_split_size': 16
    }, collate_fn=collect_mm_fn)
    