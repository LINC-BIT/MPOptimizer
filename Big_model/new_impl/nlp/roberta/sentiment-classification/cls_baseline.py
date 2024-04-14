import os
#bert_path should be the path of the roberta-base dir
os.environ['bert_path'] = '/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/nlp/roberta/sentiment-classification/roberta-base'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from methods.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md import ElasticDNN_MDPretrainingIndexAlg
from utils.dl.common.model import LayerActivation2, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
import torch.nn.functional as F
from utils.dl.common.loss import CrossEntropyLossSoft
from new_impl.cv.feat_align.main import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from new_impl.cv.feat_align.mmd import mmd_rbf
from new_impl.cv.utils.baseline_da import baseline_da
from methods.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel
from utils.common.log import logger
import json
from roberta import FMLoRA_Roberta_Util, RobertaForSenCls, FM_to_MD_Roberta_Util, ElasticRobertaUtil
from copy import deepcopy

torch.cuda.set_device(1)

# from methods.shot.shot import OnlineShotModel
from experiments.utils.elasticfm_cl import init_online_model, elasticfm_cl
# torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cuda:1'
app_name = 'secls'

scenario = build_scenario(
        source_datasets_name=['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB'],
        target_datasets_order=['HL5Domains-Nokia6610', 'HL5Domains-NikonCoolpix4300'] * 10, # TODO
        da_mode='close_set',
        data_dirs={
            **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
             for k in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                              'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']}
        },
    )

class SeClsOnlineFeatAlignModel(OnlineFeatAlignModel):
    def get_trained_params(self): # TODO: elastic fm only train a part of params
        #qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters() if 'attention.attention.projection_query' in n or 'attention.attention.projection_key' in n or 'attention.attention.projection_value' in n or 'intermediate.dense' in n or 'output.dense' in n]
        qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters()]
        return qkv_and_norm_params
    
    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'classifier'))
    
    def forward_to_get_task_loss(self, x, y):
        self.to_train_mode()
        return F.cross_entropy(self.infer(x), y)
    
    def get_mmd_loss(self, f1, f2):
        common_shape = min(f1.shape[0], f2.shape[0])
        f1 = f1.view(f1.shape[0], -1)
        f2 = f2.view(f2.shape[0], -1)
        f1 = f1[:common_shape,:]
        f2 = f2[:common_shape,:]
        return mmd_rbf(f1, f2)
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](**x)
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        _d = test_loader.dataset
        from data import build_dataloader, split_dataset
        if _d.__class__.__name__ == '_SplitDataset' and _d.underlying_dataset.__class__.__name__ == 'MergedDataset': # necessary for CL
            print('\neval on merged datasets')
            
            merged_full_dataset = _d.underlying_dataset.datasets
            ratio = len(_d.keys) / len(_d.underlying_dataset)
            
            if int(len(_d) * ratio) == 0:
                ratio = 1.
            # print(ratio)
            # bs = 
            # test_loaders = [build_dataloader(split_dataset(d, min(max(test_loader.batch_size, int(len(d) * ratio)), len(d)))[0], # TODO: this might be overlapped with train dataset
            #                                  min(test_loader.batch_size, int(len(d) * ratio)), 
            #                                  test_loader.num_workers, False, None) for d in merged_full_dataset]

            test_loaders = []
            for d in merged_full_dataset:
                n = int(len(d) * ratio)
                if n == 0:
                    n = len(d)
                sub_dataset = split_dataset(d, min(max(test_loader.batch_size, n), len(d)))[0]
                loader = build_dataloader(sub_dataset, min(test_loader.batch_size, n), test_loader.num_workers, False, None)
                test_loaders += [loader]
            
            accs = [self.get_accuracy(loader) for loader in test_loaders]
            print(accs)
            return sum(accs) / len(accs)
        
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                y = y.to(self.device)
                output = self.infer(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                # if batch_index == 0:
                #     print(pred, y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc

da_alg = FeatAlignAlg
from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
#from new_impl.cv.model import ClsOnlineFeatAlignModel
da_model = SeClsOnlineFeatAlignModel(
    app_name,
    'new_impl/nlp/roberta/sentiment-classification/results/cls_md_wo_fbs.py/20240113/999996-140353/models/md_best.pt',
    device
)

da_alg_hyp = {
    'HL5Domains-Nokia6610': {
        'train_batch_size': 32,
        'val_batch_size': 256,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 2e-7, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'feat_align_loss_weight': 1.0,
    },
    'HL5Domains-NikonCoolpix4300': {
        'train_batch_size': 32,
        'val_batch_size': 128,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 2e-7, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'feat_align_loss_weight': 1.0,
    },
}


baseline_da(
    app_name,
    scenario,
    da_alg,
    da_alg_hyp,
    da_model,
    device,
    __file__,
    "results"
)
