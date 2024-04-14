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



    
    
if __name__ == '__main__':
    from new_impl.hugging_face_impl.vit.impl import ViTHuggingFaceModelAPI
    api = ViTHuggingFaceModelAPI()

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
    # fm_model = ElasticDNN_ViT_OfflineClsFMModel('fm', fm_models_dict_path, device)
    from methods.elasticdnn.hugging_face.internal_adapter import ElasticDNN_OfflineFMModel_for_HuggingFaceFM
    fm_model = ElasticDNN_OfflineFMModel_for_HuggingFaceFM('fm', fm_models_dict_path, device)
    fm_model.set_hugging_face_api(api)
    
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
        'val_freq': 2 # NOTE: for debug, save the model every 2 iterations
    })