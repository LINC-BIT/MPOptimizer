import torch
from methods.elasticdnn.api.model import ElasticDNN_OfflineDetFMModel, ElasticDNN_OfflineDetMDModel
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.vit import FMLoRA_ViT_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.common.log import logger
import torch.nn.functional as F
import sys


class ElasticDNN_ViT_OfflineDetFMModel(ElasticDNN_OfflineDetFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        return FM_to_MD_ViT_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples)
        
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'head'), True, self.device)
    
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
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'head'), True, self.device)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        return self.models_dict['main'](x, y)['total_loss']
    
    
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    # 1. init scenario
    # scenario = build_scenario(
    #     source_datasets_name=['WI_Mask'],
    #     target_datasets_order=['MakeML_Mask'] * 10,
    #     da_mode='close_set',
    #     data_dirs={
    #         'COCO2017': '/data/zql/datasets/coco2017',
    #         'WI_Mask': '/data/zql/datasets/face_mask/WI/Medical mask/Medical mask/Medical Mask/images',
    #         'VOC2012': '/data/datasets/VOCdevkit/VOC2012/JPEGImages',
    #         'MakeML_Mask': '/data/zql/datasets/face_mask/make_ml/images'
    #     },
    # )
    scenario = build_scenario(
        source_datasets_name=['GTA5Det', 'SuperviselyPersonDet'],
        target_datasets_order=['CityscapesDet', 'BaiduPersonDet'] * 15,
        da_mode='close_set',
        data_dirs={
            'GTA5Det': '/data/zql/datasets/GTA-ls-copy/GTA5/',
            'SuperviselyPersonDet': '/data/zql/datasets/supervisely_person_full_20230635/Supervisely Person Dataset',
            'CityscapesDet': '', # not ready yet
            'BaiduPersonDet': '' # not ready yet
        },
    )
    
    # 2. init model
    device = 'cuda'
    from dnns.vit import vit_b_16
    det_model = vit_b_16(pretrained=True, num_classes=scenario.num_classes)
    from dnns.yolov3.vit_yolov3 import make_vit_yolov3
    det_model = make_vit_yolov3(det_model, torch.rand((1, 3, 224, 224)), 16, 768, scenario.num_classes, use_bigger_fpns=0, 
                                use_multi_layer_feature=False, init_head=True)
    
    fm_models_dict_path = save_models_dict_for_init({
        'main': det_model
    }, __file__, 'fm_vit_b_16_pretrained_with_det_head')
    
    fm_model = ElasticDNN_ViT_OfflineDetFMModel('fm', fm_models_dict_path, device, scenario.num_classes)
    
    # 3. init alg
    models = {
        'fm': fm_model
    }
    fm_lora_alg = ElasticDNN_FMLoRAAlg(models, get_res_save_dir(__file__, sys.argv[1]))
    
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # 4. run alg
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_lora_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': (1, 3, 224, 224),
        'ab_r': 8,
        'train_batch_size': 32,
        'val_batch_size': 16,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 310000)},
        'num_iters': 320000,
        'val_freq': 400,
        
        'fm_lora_ckpt_path': 'experiments/elasticdnn/vit_b_16/offline/fm_lora/cls/results/cls.py/20230607/999995-234355-trial/models/fm_best.pt'
    })