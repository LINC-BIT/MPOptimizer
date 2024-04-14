import torch
import sys
from torch import nn
from dnns.vit import make_softmax_prunable
from methods.elasticdnn.api.model import ElasticDNN_OfflineClsFMModel, ElasticDNN_OfflineClsMDModel
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
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

        
        
        
        
if __name__ == '__main__':
    from new_impl.hugging_face_impl.vit.impl import ViTHuggingFaceModelAPI
    api = ViTHuggingFaceModelAPI()
    
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    # 1. init model
    fm_models_dict_path = 'new_impl/hugging_face_impl/vit/offline/results/fm_lora.py/20231203/999991-170453-new_impl/hugging_face_impl/vit/offline/fm_lora.py/models/fm_best.pt'
    fm_models_dict = torch.load(fm_models_dict_path)
    fm_models_dict_path = save_models_dict_for_init(fm_models_dict, __file__, 'fm_vit_b_16_cls_lora')
    md_models_dict_path = save_models_dict_for_init({
        'main': -1
    }, __file__, 'md_vit_b_16_none')
    device = 'cuda'
    
    # fm_model = ElasticDNN_ViT_OfflineClsFMModel('fm', fm_models_dict_path, device)
    # md_model = ElasticDNN_ViT_OfflineClsMDModel('md', md_models_dict_path, device)
    
    from methods.elasticdnn.hugging_face.internal_adapter import ElasticDNN_OfflineFMModel_for_HuggingFaceFM, ElasticDNN_OfflineMDModel_for_HuggingFaceFM
    fm_model = ElasticDNN_OfflineFMModel_for_HuggingFaceFM('fm', fm_models_dict_path, device)
    fm_model.set_hugging_face_api(api)
    md_model = ElasticDNN_OfflineMDModel_for_HuggingFaceFM('md', md_models_dict_path, device)
    md_model.set_hugging_face_api(api)
    
    # 2. init alg
    models = {
        'fm': fm_model,
        'md': md_model
    }
    fm_to_md_alg = ElasticDNN_MDPretrainingWoFBSAlg(models, get_res_save_dir(__file__, sys.argv[0]))
    
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
        'generate_md_width_ratio': 4,
        
        'train_batch_size': 128,
        'val_batch_size': 512,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        'num_iters': 80000,
        'val_freq': 2,
        'distill_loss_weight': 1.0
    })
    
    # TODO:
    # 1. train MD before inserting FBS?