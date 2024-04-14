import torch
from new_impl.cv.elasticdnn.api.algs.fm_lora_glip import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from glip import FMLoRA_GLIP_Util, ElasticDNN_OfflineMMDetFMModel
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation, get_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
import torch.nn.functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class ElasticDNN_GLIP_OfflineMMDetFMModel(ElasticDNN_OfflineMMDetFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        raise NotImplementedError
        
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'visual_projection'), True, self.device), 'output'
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        raise NotImplementedError
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        # x: clip-preprocessed images and texts, y: label indexes
        x['for_training'] = True
        
        # for k, v in x.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.size())
        #     elif isinstance(v, (list, tuple)):
        #         print(k, len(v))
        #     else:
        #         print(k, v)
        
        loss_dict = self.infer(x)
        losses = sum(loss for loss in loss_dict.values())
        # print(losses)
        
        return losses
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_GLIP_Util()
    
    def get_task_head_params(self):
        return []
        
        
# class ElasticDNN_CLIP_OfflineMMClsMDModel(ElasticDNN_OfflineMMClsMDModel):
#     def get_feature_hook(self) -> LayerActivation:
#         return LayerActivation(get_module(self.models_dict['main'], 'visual_projection'), True, self.device), 'output'
    
#     def forward_to_get_task_loss(self, x, y, *args, **kwargs):
#         x['for_training'] = True
#         output = self.infer(x)
#         return output.loss
    
    
if __name__ == '__main__':
    # 1. init scenario
    scenario = build_scenario(
        source_datasets_name=['MM-COCO2017'],
        target_datasets_order=['MM-CityscapesDet'],
        da_mode='close_set',
        data_dirs={
            'MM-COCO2017': '/data/zql/datasets/coco2017',
            'MM-CityscapesDet': '/data/zql/datasets/cityscape'
        },
    )
    cfg_path = 'new_impl/cv/glip/object_detection/pretrained_model/glip_Swin_T_O365_GoldG.yaml'
    model_path = 'new_impl/cv/glip/object_detection/pretrained_model/glip_tiny_model_o365_goldg_cc_sbu.pth'
    # 2. init model
    from glip import glip_model, build_transform, run_ner, collect_mm_fn
    config, gmodel = glip_model(cfg_path, model_path)
    transform = build_transform(config, None)
    fm_models_dict_path = save_models_dict_for_init({
        'main': gmodel
    }, __file__, 'fm_glip_pretrained')
    device = 'cuda'
    
    # total_class_to_idx_map = {}
    # for v in scenario.all_datasets_e2e_class_to_idx_map.values():
    #     for k, v2 in v.items():
    #         if k in total_class_to_idx_map.keys():
    #             assert total_class_to_idx_map[k] == v2
    #         total_class_to_idx_map[k] = v2
    
    fm_model = ElasticDNN_GLIP_OfflineMMDetFMModel('fm', fm_models_dict_path, device, collate_fn=collect_mm_fn)
    
    # 3. init alg
    models = {
        'fm': fm_model
    }
    import sys
    fm_lora_alg = ElasticDNN_FMLoRAAlg(models, get_res_save_dir(__file__, tag="results"))
    
    from PIL import Image, ImageDraw
    import requests
    import numpy as np
    from evaluator import MMCOCODecoder
    ori_image = Image.open('new_impl/cv/glip/object_detection/000000103759.jpg').convert("RGB")
    image = transform(np.asarray(ori_image)[:, :, [2, 1, 0]])
    text = 'orange. umbrella. '
    targets = BoxList(torch.FloatTensor([[0., 0., 0., 0.]]), image_size=image.size()[1:], mode='xyxy')
    targets.add_field('caption', text)
    targets.add_field('tokens_positive', run_ner(text))
    targets.add_field('labels', torch.LongTensor([0]))
    samples = {'images' : [image], 'targets' : [targets]}
    
    # 4. run alg
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_lora_alg.run(scenario, hyps={
        'launch_tbboard': False,
        'transform' : transform,
        'samples_size': samples,
        'ab_r': 8,
        'train_batch_size': 16,
        'val_batch_size': 1,
        'num_workers': 16,
        'optimizer': 'Adam',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        'num_iters': 6500,
        'val_freq': 100
    }, collate_fn=collect_mm_fn)