import os
gpt_neo_series_id = '1.3B_ckpt'
os.environ['gpt_neo_series_id'] = gpt_neo_series_id
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import torch
from methods.elasticdnn.api.model import ElasticDNN_OfflineSenClsFMModel, ElasticDNN_OfflineSenClsMDModel
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from gpt_neo import FMLoRA_GPT_Util, ElasticDNN_OfflineTextGenFMModel, collate_fn
# from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
# from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
# from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_gen_scenario
from utils.common.log import logger
import torch.nn.functional as F
import sys


class ElasticDNN_GPT_OfflineTextGenFMModel(ElasticDNN_OfflineTextGenFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        # return FM_to_MD_ViT_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
        #                                                                 reducing_width_ratio, samples)
        raise NotImplementedError

    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        raise NotImplementedError
    
    def forward_to_get_task_loss(self, x, y):
        self.to_train_mode()
        return self.infer(x)
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_GPT_Util()

    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'model.lm_head')
        params_name = {k for k, v in head.named_parameters()}
        logger.info(f'task head params: {params_name}')
        return list(head.parameters())
        
        
class ElasticDNN_ViT_OfflineDetMDModel(ElasticDNN_OfflineSenClsMDModel):
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier'), True, self.device)
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        return F.cross_entropy(self.infer(x), y)
    
    
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    torch.cuda.set_device(0)
    # torch.cuda.device_count()
    # runned
    # gpt_neo_series_id = '125m_ckpt'
    # gpt_neo_series_id = '1.3B_ckpt'
    
    
    
    os.environ['gpt_neo_series_id'] = gpt_neo_series_id
    
    scenario = build_gen_scenario(
        source_datasets_name=['No_robots'],
        target_datasets_order=['No_robots'] * 1, # TODO
        da_mode='close_set',
        data_dirs={
            'No_robots': f'/data/zql/datasets/no_robots',
        },
    )
    
    # 2. init model
    device = 'cuda'
    from gpt_neo import GPTNeoForNLG, getTokenizer
    tokenizer = getTokenizer()
    model = GPTNeoForNLG(gpt_neo_series_id)
    model.model.resize_token_embeddings(len(tokenizer))
    model.model.tie_weights()
    
    fm_models_dict_path = save_models_dict_for_init({
        'main': model
    }, __file__, 'gpt_neo_pretrained_text_gen')
    
    fm_model = ElasticDNN_GPT_OfflineTextGenFMModel('fm', fm_models_dict_path, device)
    
    # 3. init alg
    models = {
        'fm': fm_model
    }
    fm_lora_alg = ElasticDNN_FMLoRAAlg(models, get_res_save_dir(__file__, "results"))
    
    
    # 4. run alg
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_lora_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': {'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]).to(device), 
                                  'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device), 
                                  'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device), 'return_dict': False},
        
        'ab_r': 8,
        'train_batch_size': 4,
        'val_batch_size': 1,
        'num_workers': 4,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 5e-5, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 80000)},
        'num_iters': 80000,
        'val_freq': 1000,
        
        # 'fm_lora_ckpt_path': 'experiments/elasticdnn/vit_b_16/offline/fm_lora/cls/results/cls.py/20230607/999995-234355-trial/models/fm_best.pt'
    }, collate_fn=collate_fn)