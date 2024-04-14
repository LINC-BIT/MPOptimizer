import torch
from new_impl.cv.elasticdnn.api.model import ElasticDNN_OfflineSegFMModel, ElasticDNN_OfflineSegMDModel
from new_impl.cv.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from new_impl.cv.elasticdnn.model.base import ElasticDNNUtil
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from sam import FMLoRA_sam_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from sam import FM_to_MD_sam_Util
from sam import ElasticsamUtil
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
import torch.nn.functional as F


class ElasticDNN_sam_OfflineSegFMModel(ElasticDNN_OfflineSegFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        return FM_to_MD_sam_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples)
        
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'head'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticsamUtil()
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(x), y)

    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_sam_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'head')
        return list(head.parameters())
        
        
class ElasticDNN_sam_OfflineSegMDModel(ElasticDNN_OfflineSegMDModel):
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'head'), True, self.device)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(x).pred_masks, y)
    
    
if __name__ == '__main__':
    # 1. init scenario
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
    
    # 2. init model\
    torch.cuda.set_device(1)
    device = 'cuda'
    # from dnns.vit import vit_b_16
    # seg_model = vit_b_16(pretrained=True, num_classes=scenario.num_classes)
    # from dnns.deeplabv3.head import DecoderLinear
    # head = DecoderLinear(scenario.num_classes, 16, 768, (224, 224)).to(device)
    # set_module(seg_model, 'head', head)
    
    # from types import MethodType
    # from timm.models.vision_transformer import VisionTransformer
    # def forward_head(self, x, pre_logits: bool = False):
    #     return self.head(x)
    # VisionTransformer.forward_head = MethodType(forward_head, seg_model)
    from sam import Sammodel
    seg_model = Sammodel.from_pretrained('new_impl/cv/sam/sam_pretrained',ignore_mismatched_sizes=True,num_classes=scenario.num_classes)
    # from dnns.deeplabv3.head import DecoderLinear
    # head = DecoderLinear(scenario.num_classes, 16, 256, (224, 224)).to(device)
    # set_module(seg_model, 'mask_decoder.iou_prediction_head', head)

    fm_models_dict_path = save_models_dict_for_init({
        'main': seg_model
    }, __file__, 'fm_sam_pretrained_with_seg_head')
    
    fm_model = ElasticDNN_sam_OfflineSegFMModel('fm', fm_models_dict_path, device, scenario.num_classes)
    
    # 3. init alg
    models = {
        'fm': fm_model
    }
    fm_lora_alg = ElasticDNN_FMLoRAAlg(models, get_res_save_dir(__file__))
    
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # 4. run alg
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_lora_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': (1, 3, 224, 224),
        'ab_r': 8,
        'train_batch_size': 16,
        'val_batch_size': 256,
        'num_workers': 16,
        'optimizer': 'Adam',
        'optimizer_args': {'lr': 5e-3, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        'num_iters': 80000,
        'val_freq': 400
    })