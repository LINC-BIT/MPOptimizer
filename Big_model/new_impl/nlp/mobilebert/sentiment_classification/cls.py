import torch
from methods.elasticdnn.api.model import ElasticDNN_OfflineSenClsFMModel, ElasticDNN_OfflineSenClsMDModel
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from bert import FMLoRA_Bert_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from bert import FM_to_MD_Bert_Util
from bert import ElasticBertUtil
from utils.dl.common.model import LayerActivation, get_module, get_parameter, set_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.common.log import logger
import torch.nn.functional as F
import sys

class ElasticDNN_BERT_OfflineClsFMModel(ElasticDNN_OfflineSenClsFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples:  torch.Tensor):
        return FM_to_MD_Bert_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples)
    
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticBertUtil()
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        pred = self.infer(x)
        
        return F.cross_entropy(pred, y)
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_Bert_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'classifier')
        params_name = {k for k, v in head.named_parameters()}
        logger.info(f'task head params: {params_name}')
        return list(head.parameters())
    
class ElasticDNN_BERT_OfflineClsMDModel(ElasticDNN_OfflineSenClsMDModel):
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier'), True, self.device)
    
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
        source_datasets_name=['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                              'HL5Domains-NikonCoolpix4300'],
        target_datasets_order=['HL5Domains-Nokia6610'] * 1, # TODO
        da_mode='close_set',
        data_dirs={
            **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
             for k in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                              'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']}
        },
    )
    
    # 2. init model
    device = 'cuda'
    from bert import bert_base_sen_cls
    cls_model = bert_base_sen_cls(num_classes=scenario.num_classes)
    # x = {'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]).to(device), 
    #      'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device), 
    #      'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device), 
    #      'return_dict': False}
    # print(cls_model(x))
    fm_models_dict_path = save_models_dict_for_init({
        'main': cls_model
    }, __file__, 'fm_bert_pretrained_with_cls_head')
    
    fm_model = ElasticDNN_BERT_OfflineClsFMModel('fm', fm_models_dict_path, device)
    # 3. init alg
    models = {
        'fm': fm_model
    }
    fm_lora_alg = ElasticDNN_FMLoRAAlg(models, get_res_save_dir(__file__, 'result'))
    
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
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
        'num_iters': 50000,
        'val_freq': 400,
        
        # 'fm_lora_ckpt_path': 'experiments/elasticdnn/vit_b_16/offline/fm_lora/cls/results/cls.py/20230607/999995-234355-trial/models/fm_best.pt'
    })