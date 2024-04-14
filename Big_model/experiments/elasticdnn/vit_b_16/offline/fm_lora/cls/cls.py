import torch
from methods.elasticdnn.api.model import ElasticDNN_OfflineClsFMModel, ElasticDNN_OfflineClsMDModel
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.vit import FMLoRA_ViT_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation, get_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
import torch.nn.functional as F


class ElasticDNN_ViT_OfflineClsFMModel(ElasticDNN_OfflineClsFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        return FM_to_MD_ViT_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples)
        
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'head'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticViTUtil()
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        #print(self.infer(x))
        #print(self.models_dict['main']))
        #print(F.cross_entropy(self.infer(x), y))
        return F.cross_entropy(self.infer(x), y)
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_ViT_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'head')
        return list(head.parameters())
        
        
class ElasticDNN_ViT_OfflineClsMDModel(ElasticDNN_OfflineClsMDModel):
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'head'), True, self.device)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
       
        return F.cross_entropy(self.infer(x), y)
    
    
if __name__ == '__main__':
    # 1. init scenario
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
    
    # 2. init model
    from dnns.vit import vit_b_16
    fm_models_dict_path = save_models_dict_for_init({
        'main': vit_b_16(pretrained=True, num_classes=scenario.num_classes)
    }, __file__, 'fm_vit_b_16_pretrained')
    device = 'cuda'
    fm_model = ElasticDNN_ViT_OfflineClsFMModel('fm', fm_models_dict_path, device)
    
    # 3. init alg
    models = {
        'fm': fm_model
    }
    import sys
    fm_lora_alg = ElasticDNN_FMLoRAAlg(models, get_res_save_dir(__file__, tag=sys.argv[0]))
    
    # 4. run alg
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_lora_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': (1, 3, 224, 224),
        'ab_r': 8,
        'train_batch_size': 256,
        'val_batch_size': 512,
        'num_workers': 16,
        'optimizer': 'Adam',
        'optimizer_args': {'lr': 5e-4, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        'num_iters': 80000,
        'val_freq': 4000
    })