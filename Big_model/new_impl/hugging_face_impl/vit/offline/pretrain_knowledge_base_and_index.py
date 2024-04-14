import torch
import sys
from torch import nn
from dnns.vit import make_softmax_prunable
from methods.elasticdnn.api.model import ElasticDNN_OfflineClsFMModel, ElasticDNN_OfflineClsMDModel
# from methods.elasticdnn.api.algs.md_pretraining_w_fbs import ElasticDNN_MDPretrainingWFBSAlg
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
    fm_models_dict_path = 'new_impl/hugging_face_impl/vit/offline/results/pretrain_knowledge_base.py/20231203/999991-172715-new_impl/hugging_face_impl/vit/offline/pretrain_knowledge_base.py/models/fm_best.pt'
    fm_models_dict_path = save_models_dict_for_init(torch.load(fm_models_dict_path), __file__, 'fm_vit_b_16_cls_lora')
    pretrained_md_models_dict_path = 'new_impl/hugging_face_impl/vit/offline/results/pretrain_knowledge_base.py/20231203/999991-172715-new_impl/hugging_face_impl/vit/offline/pretrain_knowledge_base.py/models/md_best.pt'
    md_models_dict = torch.load(pretrained_md_models_dict_path)
    md_models_dict_path = save_models_dict_for_init(md_models_dict, __file__, 'md_vit_b_16_cls_pretrained_wo_fbs')
    device = 'cuda'
    
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
    from methods.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md import ElasticDNN_MDPretrainingIndexAlg

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
        'optimizer_args': {'lr': 1e-5, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        
        'max_sparsity': 0.9,
        'min_sparsity': 0.0,
        'num_iters': 60000,
        'val_freq': 2,
        'index_loss_weight': 1e-4,
        'l1_reg_loss_weight': 1e-9,
        'val_num_sparsities': 4,
        
        'bn_cal_num_iters': 0,
        
        'index_init': 'zero',
        'index_guided_linear_comb_split_size': 512
    })
    