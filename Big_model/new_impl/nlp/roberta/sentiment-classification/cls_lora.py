import os
#bert_path should be the path of the roberta-base dir
os.environ['bert_path'] = '/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/nlp/roberta/sentiment-classification/roberta-base'

import torch
from methods.elasticdnn.api.model import ElasticDNN_OfflineSenClsFMModel, ElasticDNN_OfflineSenClsMDModel
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
# from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
# from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
# from methods.elasticdnn.model.vit import ElasticViTUtil
from roberta import FMLoRA_Roberta_Util, RobertaForSenCls
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.common.log import logger
import torch.nn.functional as F
import sys

class ElasticDNN_Roberta_OfflineSenClsFMModel(ElasticDNN_OfflineSenClsFMModel):
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
        pred = self.infer(x)
        return F.cross_entropy(pred, y)
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_Roberta_Util()

    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'classifier')
        params_name = {k for k, v in head.named_parameters()}
        logger.info(f'task head params: {params_name}')
        return list(head.parameters())
    
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    torch.cuda.set_device(1)

    scenario = build_scenario(
        source_datasets_name=['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB'],
        target_datasets_order=['HL5Domains-Nokia6610', 'HL5Domains-NikonCoolpix4300'] * 1, # TODO
        da_mode='close_set',
        data_dirs={
            **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
             for k in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                              'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']}
        },
    )
    
    # 2. init model
    device = 'cuda'
    model = RobertaForSenCls(num_classes=scenario.num_classes)
    fm_models_dict_path = save_models_dict_for_init({
        'main': model
    }, __file__, 'roberta_pretrained_sen_cls')
    
    fm_model = ElasticDNN_Roberta_OfflineSenClsFMModel('fm', fm_models_dict_path, device)
    
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
        'train_batch_size': 32,
        'val_batch_size': 128,
        'num_workers': 32,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 80000)},
        'num_iters': 80000,
        'val_freq': 1000,
    })