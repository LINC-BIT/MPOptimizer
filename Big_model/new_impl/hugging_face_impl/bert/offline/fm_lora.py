import torch
from methods.elasticdnn.api.model import ElasticDNN_OfflineSenClsFMModel, ElasticDNN_OfflineSenClsMDModel
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.bert import FMLoRA_Bert_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.common.log import logger
import torch.nn.functional as F
import sys

    
if __name__ == '__main__':
    from new_impl.hugging_face_impl.bert.impl import BERTHuggingFaceModelAPI
    api = BERTHuggingFaceModelAPI()
    
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    scenario = build_scenario(
        source_datasets_name=['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                              'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610'],
        target_datasets_order=['Liu3Domains-Computer'] * 1, # TODO
        da_mode='close_set',
        data_dirs={
            **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
             for k in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                              'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']},
            'Liu3Domains-Computer': ''
        },
    )
    
    # 2. init model
    device = 'cuda'
    from dnns.bert import bert_base_sen_cls
    model = bert_base_sen_cls(num_classes=scenario.num_classes)
    from methods.elasticdnn.pipeline.fm_to_md.bert import BertSelfAttentionPrunable
    for block in model.bert.encoder.layer:
        set_module(block, 'attention.self', BertSelfAttentionPrunable.init_from_exist_self_attn(block.attention.self))
    
    fm_models_dict_path = save_models_dict_for_init({
        'main': model
    }, __file__, 'fm_bert_base_pretrained_with_sen_cls_head')
    
    # fm_model = ElasticDNN_Bert_OfflineSenClsFMModel('fm', fm_models_dict_path, device)
    from methods.elasticdnn.hugging_face.internal_adapter import ElasticDNN_OfflineFMModel_for_HuggingFaceFM
    fm_model = ElasticDNN_OfflineFMModel_for_HuggingFaceFM('fm', fm_models_dict_path, device)
    fm_model.set_hugging_face_api(api)
    
    # 3. init alg
    models = {
        'fm': fm_model
    }
    fm_lora_alg = ElasticDNN_FMLoRAAlg(models, get_res_save_dir(__file__, sys.argv[1]))
    
    
    # 4. run alg
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_lora_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': {'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]).to(device), 
                                  'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device), 
                                  'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device), 'return_dict': False},
        
        'ab_r': 8,
        'train_batch_size': 8,
        'val_batch_size': 16,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 310000)},
        'num_iters': 320000,
        'val_freq': 2,
        
        # 'fm_lora_ckpt_path': 'experiments/elasticdnn/vit_b_16/offline/fm_lora/cls/results/cls.py/20230607/999995-234355-trial/models/fm_best.pt'
    })