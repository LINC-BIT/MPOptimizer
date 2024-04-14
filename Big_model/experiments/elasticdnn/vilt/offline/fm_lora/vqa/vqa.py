import torch
from methods.elasticdnn.api.model import ElasticDNN_OfflineVQAFMModel, ElasticDNN_OfflineVQAMDModel
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.vilt import FMLoRA_Vilt_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.common.log import logger
import torch.nn.functional as F
import sys


class ElasticDNN_Vilt_OfflineVQAFMModel(ElasticDNN_OfflineVQAFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        # return FM_to_MD_ViT_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
        #                                                                 reducing_width_ratio, samples)
        raise NotImplementedError

    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier.3'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        raise NotImplementedError
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        # print(x['input_ids'].size(), x['pixel_values'].size(), )
        o = self.infer(x).logits
        
        # print(o.size(), y.size(), o, y)
        
        return F.binary_cross_entropy_with_logits(o, y) * y.shape[1]
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_Vilt_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'classifier')
        params_name = {k for k, v in head.named_parameters()}
        logger.info(f'task head params: {params_name}')
        return list(head.parameters())
        
        
class ElasticDNN_Vilt_OfflineVQAMDModel(ElasticDNN_OfflineVQAMDModel):
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier.3'), True, self.device)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        o = self.infer(x)
        return nn.functional.binary_cross_entropy_with_logits(o, y) * y.shape[1]
    
    
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    scenario = build_scenario(
        source_datasets_name=['VQAv2_split1'],
        target_datasets_order=['VQAv2_split1_c'] * 1, # TODO
        da_mode='close_set',
        data_dirs={
            'VQAv2_split1': '/data/zql/datasets/vqav2',
            'VQAv2_split1_c': '/data/zql/datasets/vqav2'
        },
    )
    
    # 2. init model
    device = 'cuda'
    from dnns.vilt import vilt_b_32
    model = vilt_b_32(num_classes=scenario.num_classes)
    
    fm_models_dict_path = save_models_dict_for_init({
        'main': model
    }, __file__, 'fm_vilt')
    
    fm_model = ElasticDNN_Vilt_OfflineVQAFMModel('fm', fm_models_dict_path, device)
    
    # 3. init alg
    models = {
        'fm': fm_model
    }
    fm_lora_alg = ElasticDNN_FMLoRAAlg(models, get_res_save_dir(__file__, sys.argv[0]))
    
    
    sample_dataset = list(scenario.get_offline_datasets().values())[0]['train']
    sample = sample_dataset[0][0]
    
    for k, v in sample.items():
        sample[k] = v.unsqueeze(0)
        
    # 4. run alg
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_lora_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': sample,
        
        'ab_r': 8,
        'train_batch_size': 64,
        'val_batch_size': 512,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 310000)},
        'num_iters': 320000,
        'val_freq': 400,
        
        # 'fm_lora_ckpt_path': 'experiments/elasticdnn/vit_b_16/offline/fm_lora/cls/results/cls.py/20230607/999995-234355-TokenClsial/models/fm_best.pt'
    })