import torch
import sys
from torch import nn
from methods.elasticdnn.api.model import ElasticDNN_OfflineTokenClsFMModel, ElasticDNN_OfflineTokenClsMDModel
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.bert import FM_to_MD_Bert_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_lora.bert import FMLoRA_Bert_Util
from methods.elasticdnn.model.bert import ElasticBertUtil
from utils.dl.common.model import LayerActivation2, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
from utils.dl.common.loss import CrossEntropyLossSoft
import torch.nn.functional as F
from utils.common.log import logger


class ElasticDNN_Bert_OfflineTokenClsFMModel(ElasticDNN_OfflineTokenClsFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor): # TODO:
        return FM_to_MD_Bert_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
                                                                        reducing_width_ratio, samples)
        # raise NotImplementedError

    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'classifier'))
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil: # TODO:
        return ElasticBertUtil()
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        o = self.infer(x)
        return F.cross_entropy(o.view(-1, o.size(-1)), y.view(-1))
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_Bert_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'classifier')
        params_name = {k for k, v in head.named_parameters()}
        logger.info(f'task head params: {params_name}')
        return list(head.parameters())
        
        
class ElasticDNN_Bert_OfflineTokenClsMDModel(ElasticDNN_OfflineTokenClsMDModel):
    def __init__(self, name: str, models_dict_path: str, device: str):
        super().__init__(name, models_dict_path, device)
        
        self.distill_criterion = CrossEntropyLossSoft()
        
    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'classifier'))
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        self.to_train_mode()
        o = self.infer(x)
        return F.cross_entropy(o.view(-1, o.size(-1)), y.view(-1))
    
    def get_distill_loss(self, student_output, teacher_output):
        # print(student_output, teacher_output)
        return self.distill_criterion(student_output.view(-1, student_output.size(-1)), teacher_output.view(-1, teacher_output.size(-1)))
    
    def get_matched_param_of_fm(self, self_param_name, fm: nn.Module): # TODO:
        if any([k in self_param_name for k in ['fbs', 'ab', 'embeddings']]):
            return None
        
        p = get_parameter(self.models_dict['main'], self_param_name)
        if p.dim() == 0:
            return None
        
        elif p.dim() == 1 and 'LayerNorm' in self_param_name and 'weight' in self_param_name:
            # if self_param_name.startswith('norm'):
            #     return None
            return get_parameter(fm, self_param_name)
        
        # 1. xx.query.weight -> xx.query.fc.weight and xx.query.ab.0/1
        if ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('weight' in self_param_name):
            ss = self_param_name.split('.')
            # raise NotImplementedError() # TODO:
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs[1].weight @ fm_abs[0].weight
            ], dim=0)
            
        elif ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('bias' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc.bias'
            return get_parameter(fm, fm_qkv_name)
            
        elif 'intermediate.dense' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)

        else:
            return get_parameter(fm, self_param_name)
        
    
     
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    # 3. init scenario
    scenario = build_scenario(
        source_datasets_name=['HL5Domains-ApexAD2600Progressive-TokenCls', 'HL5Domains-CanonG3-TokenCls', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB-TokenCls',
                              'HL5Domains-NikonCoolpix4300-TokenCls', 'HL5Domains-Nokia6610-TokenCls'],
        target_datasets_order=['Liu3Domains-Computer-TokenCls'] * 1, # TODO
        da_mode='close_set',
        data_dirs={
            **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
             for k in ['HL5Domains-ApexAD2600Progressive-TokenCls', 'HL5Domains-CanonG3-TokenCls', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB-TokenCls',
                              'HL5Domains-NikonCoolpix4300-TokenCls', 'HL5Domains-Nokia6610-TokenCls']},
            'Liu3Domains-Computer-TokenCls': ''
        },
    )
    
    # 1. init model
    # from dnns.deeplabv3.head import modify_forward_head
    # modify_forward_head() # TODO: bring a bug
    # from dnns.vit import vit_b_16
    fm_models_dict_path = 'experiments/elasticdnn/bert_base/offline/fm_to_md/pos/results/pos_md_wo_fbs.py/20230703/999997-181910/models/fm_best.pt'
    fm_models = torch.load(fm_models_dict_path)
    fm_models_dict_path = save_models_dict_for_init(fm_models, __file__, 'fm_bert_base_secls_lora')
    md_models_dict_path = save_models_dict_for_init(
        torch.load('experiments/elasticdnn/bert_base/offline/fm_to_md/pos/results/pos_md_wo_fbs.py/20230703/999997-181910/models/md_best.pt'), 
        __file__, 'md_bert_base_secls_raw_pretrained')
    device = 'cuda'
    
    fm_model = ElasticDNN_Bert_OfflineTokenClsFMModel('fm', fm_models_dict_path, device)
    md_model = ElasticDNN_Bert_OfflineTokenClsMDModel('md', md_models_dict_path, device)
    
    # 2. init alg
    models = {
        'fm': fm_model,
        'md': md_model
    }
    from methods.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md import ElasticDNN_MDPretrainingIndexAlg
    fm_to_md_alg = ElasticDNN_MDPretrainingIndexAlg(models, get_res_save_dir(__file__, sys.argv[1]))
    
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_to_md_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': {'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]]).to(device), 
                                  'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device), 
                                  'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device), 'return_dict': False},
        
        'FBS_r': 16,
        'FBS_ignore_layers': [],
        
        'train_batch_size': 8,
        'val_batch_size': 16,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 1e-4, 'betas': [0.9, 0.999]},
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        
        'indexes_optimizer_args': {'lr': 3e-3, 'momentum': 0.9, 'weight_decay': 5e-4},
        
        'num_iters': 80000,
        'val_freq': 400,
        
        'max_sparsity': 0.9,
        'min_sparsity': 0.0,
        'l1_reg_loss_weight': 1e-9,
        'index_loss_weight': 1e-4,
        'val_num_sparsities': 4,
        
        'bn_cal_num_iters': 0,
        
        'index_init': 'zero',
        'index_guided_linear_comb_split_size': 512
    })
    