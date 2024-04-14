import os
gpt_neo_series_id = '1.3B_ckpt'
os.environ['gpt_neo_series_id'] = gpt_neo_series_id
import torch
import torch.nn as nn
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from gpt_neo import getTokenizer, ElasticGPTUtil, FMLoRA_GPT_Util, ElasticDNN_OfflineTextGenFMModel, ElasticDNN_OfflineTextGenMDModel, FM_to_MD_GPT_Util, collate_fn
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from methods.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md import ElasticDNN_MDPretrainingIndexAlg
from utils.dl.common.model import LayerActivation2, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_gen_scenario
import torch.nn.functional as F
import os
from utils.dl.common.loss import CrossEntropyLossSoft
from new_impl.cv.feat_align.main_gpt_neo import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from new_impl.cv.feat_align.mmd import mmd_rbf
from new_impl.cv.utils.baseline_da import baseline_da
from new_impl.cv.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel
from utils.common.log import logger
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import json


os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(1)
device = 'cuda'
app_name = 'cls'

scenario = build_gen_scenario(
        source_datasets_name=['No_robots'],
        target_datasets_order=['Medicine_task', 'Law_task'] * 10,
        da_mode='close_set',
        data_dirs={
            'No_robots': '/data/zql/datasets/no_robots',
            'Law_task': '/data/zql/datasets/law_task',
            'Medicine_task': '/data/zql/datasets/medicine_task',
        },
    )
    
class TxtgenOnlineFeatAlignModel(OnlineFeatAlignModel):
    def get_trained_params(self): # TODO: elastic fm only train a part of params
        #qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters() if 'attention.attention.projection_query' in n or 'attention.attention.projection_key' in n or 'attention.attention.projection_value' in n or 'intermediate.dense' in n or 'output.dense' in n]
        qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters()]
        return qkv_and_norm_params
    
    def get_feature_hook(self) -> LayerActivation2:
        return LayerActivation2(get_module(self.models_dict['main'], 'model.lm_head'))
    
    def forward_to_get_task_loss(self, x, y):
        losses = self.infer(x)
        # print(losses)
        
        return losses
    
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
        acc = 0
        sample_num = 0
        tokenizer = getTokenizer()
        self.to_eval_mode()
        pred_txt = []
        true_txt = []
        res = []
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, _) in pbar:
                if len(x) == 0:
                    continue
                # if batch_index > 10:
                #     break
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.device)
                # input_ids = []
                inputlen = x['len']
                y = x['labels']
                x['labels'] = None
                outputs = self.models_dict['main'].generate(x, pad_id=tokenizer.pad_token_id)
                
                for i, op in enumerate(outputs):
                    op = op.tolist()
                    op = list(filter(lambda x: x != tokenizer.pad_token_id, op))
                    txt = tokenizer.decode(op)
                    txt = txt.replace(tokenizer.pad_token, "")
                    res.append(txt)
                    txt = txt[inputlen[i]:]
                    pred_txt.append(nltk.word_tokenize(txt))
                for tp in y:
                    true_txt.append(nltk.word_tokenize(tokenizer.decode(tp).replace(tokenizer.pad_token, '')))
                # pred = F.softmax(output, dim=1).argmax(dim=1)
                # correct = torch.eq(pred, y).sum().item()
                # acc += correct
                sample_num += len(y)
                
                # pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                #                      f'cur_batch_acc: {(correct / len(y)):.4f}')
        json.dump(res, open("./gpt_generation.json", "w"))
        smooth = SmoothingFunction()
        score = 0.
        for pred, true in zip(pred_txt, true_txt):
            score += sentence_bleu([true], pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        score /= sample_num
        return score

da_alg = FeatAlignAlg
from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
#from new_impl.cv.model import ClsOnlineFeatAlignModel
da_model = TxtgenOnlineFeatAlignModel(
    app_name,
    'new_impl/nlp/gpt-neo/text_generation/results/gen_md_wo_fbs.py/20240113/999999-172009/models/md_best.pt',
    device
)
da_alg_hyp = {
    'Medicine_task': {
        'train_batch_size': 2,
        'val_batch_size': 1,
        'num_workers': 2,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 5e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 1000,
        'val_freq': 200,
        'feat_align_loss_weight': 1.0,
    },
    'Law_task': {
        'train_batch_size': 2,
        'val_batch_size': 1,
        'num_workers': 2,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 5e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 1000,
        'val_freq': 200,
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
    "results",
    collate_fn=collate_fn
)