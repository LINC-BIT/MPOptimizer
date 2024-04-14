import torch
from torch import nn
from copy import deepcopy
from abc import ABC, abstractmethod
from methods.elasticdnn.api.model import ElasticDNN_OfflineFMModel, ElasticDNN_OfflineMDModel
from utils.dl.common.model import get_model_device, get_model_latency, get_model_size, set_module, get_module
from utils.common.log import logger
from transformers import GPTNeoForCausalLM
from utils.dl.common.model import set_module
from torch import nn
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.common.log import logger
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util, LoRA
from methods.elasticdnn.model.base import Abs, KTakesAll, ElasticDNNUtil, Layer_WrappedWithFBS
import tqdm
from transformers import GPT2Tokenizer
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import json

def collate_fn(batch):
    dict = {}
    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []
    return_dict = True
    lenli = []
    for item in batch:
        if len(item) == 1 or len(item['labels']) == 0:
            continue
        input_ids.append(item['input_ids'].unsqueeze(0))
        if 'attention_mask' in item.keys():
            attention_mask.append(item['attention_mask'].unsqueeze(0))
        if 'token_type_ids' in item.keys():
            token_type_ids.append(item['token_type_ids'].unsqueeze(0))
        labels.append(item['labels'].unsqueeze(0))
        if 'len' in item.keys():
            lenli.append(item['len'])

    dict['return_dict'] = batch[0]['return_dict']
    if len(input_ids) > 0:
        dict['input_ids'] = torch.cat(input_ids, dim=0)
    else:
        return {}, torch.Tensor([0])
    if len(attention_mask) > 0:
        dict['attention_mask'] = torch.cat(attention_mask, dim=0)
    if len(token_type_ids) > 0:
        dict['token_type_ids'] = torch.cat(token_type_ids, dim=0)
    dict['labels'] = torch.cat(labels, dim=0)
    if len(lenli) > 0:
        dict['len'] = lenli
    return dict, torch.Tensor([0])

class GPTNeoForNLG(nn.Module):
    def __init__(self, series_id):
        super(GPTNeoForNLG, self).__init__()
        
        # logger.info(f'init bert for sen cls (using {bert_model_tag})')
        self.model = GPTNeoForCausalLM.from_pretrained(f'experiments/elasticdnn/gpt_neo/{series_id}')
        self.config = self.model.config
        
        self.config.pad_token_id = self.config.eos_token_id
    
    def generate(self, x, pad_id=None):
        return self.model.generate(x['input_ids'], max_new_tokens=128, num_beams=3, early_stopping=True, pad_token_id=pad_id)
        
    def forward(self, **x):
        x['return_dict'] = True

        output = self.model(**x)
        
        if self.training == True:
            return output['loss']
        else:
            return output['logits']

class ToQKV_WrappedWithLoRA(nn.Module):
    def __init__(self, fc: nn.Linear, ab_r: int):
        super(ToQKV_WrappedWithLoRA, self).__init__()
        
        self.fc = fc
        self.ab = self.create_ab_as_linear(fc.weight.data, ab_r)
        
    def create_ab_as_linear(self, fc_weight: torch.Tensor, ab_r: int):
        res = nn.Sequential(
            LoRA(fc_weight.size(1), fc_weight.size(0) // ab_r, bias=False),
            LoRA(fc_weight.size(0) // ab_r, fc_weight.size(0), bias=False)
        ).to(fc_weight.device)
        nn.init.kaiming_uniform_(res[0].weight, a=5 ** 0.5)
        nn.init.zeros_(res[1].weight)
        return res
        
    def forward(self, x):
        x1 = self.fc(x)
        x2 = self.ab(x)
        return x1 + x2
    

class FMLoRA_GPT_Util(FMLoRA_Util):
    
    @torch.no_grad()
    def add_lora_ab_to_fm(self, fm: nn.Module, ab_r: int, samples: dict):
        fm.eval()
        
        o1 = fm(**samples)
        
        for name, module in fm.named_modules():
            if name.endswith(('k_proj', 'v_proj', 'q_proj')):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
        
        o2 = fm(**samples)
        
        if isinstance(o1, tuple):
            o1 = o1[-1]
            o2 = o2[-1]
        output_diff = ((o1 - o2) ** 2).sum()
        assert output_diff < 1e-5
        
        return fm
    
    @torch.no_grad()
    def absorb_lora_and_recover_net_structure(self, fm: nn.Module, samples: dict):       
        fm.eval()
        # print('absorb lora before')
        o1 = fm(**samples)
        
        for name, module in fm.named_modules():
            if not isinstance(module, ToQKV_WrappedWithLoRA):
                continue
            
            fc = module.fc
            ab = module.ab

            fc.weight.add_(ab[1].weight @ ab[0].weight)
            
            set_module(fm, name, fc)
        
        # print('absorb lora after')
        o2 = fm(**samples)
        
        if isinstance(o1, tuple):
            o1 = o1[-1]
            o2 = o2[-1]
        output_diff = ((o1 - o2) ** 2).sum()
        # assert output_diff < 1e-6, output_diff
        
        return fm

def getTokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(f'experiments/elasticdnn/gpt_neo/{os.environ["gpt_neo_series_id"]}', padding_side='left')
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token
    special_tokens = {"pad_token":"<|pad|>"}#, "sep_token":"<|sep|>", "bos_token":"<|bos|>"}
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = "<|pad|>"
    # tokenizer.bos_token = "<|bos|>"
    # tokenizer.sep_token = "<|sep|>"
    return tokenizer

class ElasticDNN_OfflineTextGenFMModel(ElasticDNN_OfflineFMModel):
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
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'].forward(**x)

class FM_to_MD_GPT_Util(FM_to_MD_Util):
    def init_md_from_fm_by_reducing_width(self, fm: nn.Module, reducing_width_ratio: int, sparsity=0.0) -> nn.Module:
        # sparsity: It is mainly used to make a distilled model used in the baseline algorithm, and this parameter can ensure that the model has the same size as the model used in the online algorithm.
        fm_vit = deepcopy(fm)
        
        for block in fm_vit.model.transformer.h:
            tmp = get_module(block, 'attn.attention')
            tmp.head_dim = max(tmp.head_dim // reducing_width_ratio, 16)
            set_module(block, 'attn.attention', tmp)
        #ddw
        def _f(n):
            return int(n // reducing_width_ratio)
        
        # def _rand_indexes(n):
            # return torch.randperm(n)[0: int(n // reducing_width_ratio)]
        
        def l1_max_indexes_for_qkv(p: torch.Tensor, dim=0):
            assert dim in [0, 1]
            assert p.dim() in [1, 2, 4]
            
            if dim == 1:
                p = p.T
            
            p_norm = p.abs().contiguous().view(p.size(0), -1).sum(dim=1)
            n = p.size(0)
            return p_norm.argsort(descending=True)[0: max(256, int(n // reducing_width_ratio))].sort()[0]

        def l1_max_indexes(p: torch.Tensor, dim=0):
            assert dim in [0, 1]
            assert p.dim() in [1, 2, 4]
            
            if dim == 1:
                p = p.T
            
            p_norm = p.abs().contiguous().view(p.size(0), -1).sum(dim=1)
            n = p.size(0)
            return p_norm.argsort(descending=True)[0: int(n // reducing_width_ratio)].sort()[0]
        
        def l1_max_indexes_with_sparsity(p: torch.Tensor, dim=0):
            assert dim in [0, 1]
            assert p.dim() in [1, 2, 4]
            
            if dim == 1:
                p = p.T
            
            p_norm = p.abs().contiguous().view(p.size(0), -1).sum(dim=1)
            n = p.size(0)
            return p_norm.argsort(descending=True)[0: int(n // reducing_width_ratio * (1 - sparsity))].sort()[0]

        for block_i, block in enumerate(fm_vit.model.transformer.h):
            for k in ['k_proj', 'v_proj', 'q_proj']:
                qkv = get_module(block, f'attn.attention.{k}')

                new_qkv = nn.Linear(qkv.in_features, max(256, _f(qkv.out_features)), 
                                    qkv.bias is not None, qkv.weight.device)
                indexes = l1_max_indexes_for_qkv(qkv.weight.data, 0)
                
                new_qkv.weight.data.copy_(qkv.weight.data[indexes])
                if qkv.bias is not None:
                    new_qkv.bias.data.copy_(qkv.bias.data[indexes])
                set_module(block, f'attn.attention.{k}', new_qkv)
            
            proj = get_module(block, f'attn.attention.out_proj')
            new_proj = nn.Linear(max(256, _f(proj.in_features)), proj.out_features, 
                                proj.bias is not None, proj.weight.device)
            new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes_for_qkv(proj.weight.data, 1)])
            if proj.bias is not None:
                new_proj.bias.data.copy_(proj.bias.data)
            set_module(block, f'attn.attention.out_proj', new_proj)
            
            fc1 = get_module(block, f'mlp.c_fc')
            new_fc1 = nn.Linear(fc1.in_features, int(_f(fc1.out_features) * (1 - sparsity)), 
                                fc1.bias is not None, fc1.weight.device)
            indexes = l1_max_indexes_with_sparsity(fc1.weight.data, 0)
            new_fc1.weight.data.copy_(fc1.weight.data[indexes])
            if fc1.bias is not None:
                new_fc1.bias.data.copy_(fc1.bias.data[indexes])
            set_module(block, f'mlp.c_fc', new_fc1)

            fc2 = get_module(block, f'mlp.c_proj')
            new_fc2 = nn.Linear(int(_f(fc2.in_features) * (1 - sparsity)), fc2.out_features, 
                                fc2.bias is not None, fc2.weight.device)
            new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes_with_sparsity(fc2.weight.data, 1)])
            if fc2.bias is not None:
                new_fc2.bias.data.copy_(fc2.bias.data)
            set_module(block, f'mlp.c_proj', new_fc2)
            
        return fm_vit
    
    def init_md_from_fm_by_reducing_width_with_perf_test(self, fm: nn.Module, reducing_width_ratio: int,
                                                         samples: torch.Tensor) -> nn.Module:
        fm_size = get_model_size(fm, True)
        fm_latency = self._get_model_latency(fm, samples, 20, 
                                               get_model_device(fm), 20, False)
        
        master_dnn = self.init_md_from_fm_by_reducing_width(fm, reducing_width_ratio)
        master_dnn_size = get_model_size(master_dnn, True)
        logger.debug(f'inited master DNN: {master_dnn}')
        master_dnn_latency = self._get_model_latency(master_dnn, samples, 20, 
                                               get_model_device(master_dnn), 20, False)

        logger.info(f'init master DNN (w/o FBS yet) by reducing foundation model\'s width (by {reducing_width_ratio:d}x)')
        logger.info(f'foundation model ({fm_size:.3f}MB, {fm_latency:.4f}s/sample) -> '
                    f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(fm_size / master_dnn_size):.2f}x, '
                    f'latency: ↓ {(fm_latency / master_dnn_latency):.2f}x)')
        
        return master_dnn
    
    def init_md_from_fm_by_reducing_layers(self, fm: nn.Module, layers: list) -> nn.Module:
        fm_vit = deepcopy(fm)
        tmp_h = []
        for block_i, block in enumerate(fm_vit.model.transformer.h):
            if block_i in layers:
                tmp_h.append(block)
        
        tmp_h = nn.ModuleList(tmp_h)
        set_module(fm_vit, f'model.transformer.h', tmp_h)
            
        return fm_vit
    
    def init_md_from_fm_by_reducing_layers_with_perf_test(self, fm: nn.Module, layers: list,
                                                         samples: torch.Tensor) -> nn.Module:
        fm_size = get_model_size(fm, True)
        fm_latency = self._get_model_latency(fm, samples, 20, 
                                               get_model_device(fm), 20, False)
        
        master_dnn = self.init_md_from_fm_by_reducing_layers(fm, layers)
        master_dnn_size = get_model_size(master_dnn, True)
        logger.debug(f'inited master DNN: {master_dnn}')
        master_dnn_latency = self._get_model_latency(master_dnn, samples, 20, 
                                               get_model_device(master_dnn), 20, False)

        logger.info(f'init master DNN (w/o FBS yet) by reducing foundation model\'s layers (to {len(layers):d} layers)')
        logger.info(f'foundation model ({fm_size:.3f}MB, {fm_latency:.4f}s/sample) -> '
                    f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(fm_size / master_dnn_size):.2f}x, '
                    f'latency: ↓ {(fm_latency / master_dnn_latency):.2f}x)')
        
        return master_dnn

    def _get_model_latency(self, model: torch.nn.Module, model_input_size, sample_num: int, 
                           device: str, warmup_sample_num: int, return_detail=False):
        import time
        
        if isinstance(model_input_size, tuple):
            dummy_input = torch.rand(model_input_size).to(device)
        else:
            dummy_input = model_input_size
            
        model = model.to(device)
        model.eval()
        
        # warm up
        with torch.no_grad():
            for _ in range(warmup_sample_num):
                model(**dummy_input)
                
        infer_time_list = []
                
        if device == 'cuda' or 'cuda' in str(device):
            with torch.no_grad():
                for _ in range(sample_num):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    model(**dummy_input)
                    e.record()
                    torch.cuda.synchronize()
                    cur_model_infer_time = s.elapsed_time(e) / 1000.
                    infer_time_list += [cur_model_infer_time]

        else:
            with torch.no_grad():
                for _ in range(sample_num):
                    start = time.time()
                    model(**dummy_input)
                    cur_model_infer_time = time.time() - start
                    infer_time_list += [cur_model_infer_time]
                    
        avg_infer_time = sum(infer_time_list) / sample_num

        if return_detail:
            return avg_infer_time, infer_time_list
        return avg_infer_time

class ElasticDNN_OfflineTextGenMDModel(ElasticDNN_OfflineMDModel):
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
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'].forward(**x)

class SqueezeLast(nn.Module):
    def __init__(self):
        super(SqueezeLast, self).__init__()
    
    def forward(self, x):
        return x.squeeze(-1)

class StaticFBS(nn.Module):
    def __init__(self, static_channel_attention):
        super(StaticFBS, self).__init__()
        assert static_channel_attention.dim() == 2 and static_channel_attention.size(0) == 1
        self.static_channel_attention = nn.Parameter(static_channel_attention, requires_grad=False) # (1, dim)
        
    def forward(self, x):
        # print('staticfbs', x, self.static_channel_attention.unsqueeze(1))
        return x * self.static_channel_attention.unsqueeze(1)

class Linear_WrappedWithFBS(Layer_WrappedWithFBS):
    def __init__(self, linear: nn.Linear, r):
        super(Linear_WrappedWithFBS, self).__init__()
        
        self.linear = linear
        
        # for conv: (B, C_in, H, W) -> (B, C_in) -> (B, C_out)
        # for mlp in ViT: (B, #patches, D: dim of patches embedding) -> (B, D) -> (B, C_out)
        self.fbs = nn.Sequential(
            Rearrange('b n d -> b d n'),
            Abs(),
            nn.AdaptiveAvgPool1d(1),
            SqueezeLast(),
            nn.Linear(linear.in_features, max(linear.out_features // r, 36)),
            nn.ReLU(),
            nn.Linear(max(linear.out_features // r, 36), linear.out_features),
            nn.ReLU()
        )
        
        nn.init.constant_(self.fbs[6].bias, 1.)
        nn.init.kaiming_normal_(self.fbs[6].weight)
        
    
    def forward(self, x):
        if self.use_cached_channel_attention and self.cached_channel_attention is not None:
            channel_attention = self.cached_channel_attention
        else:
            self.cached_raw_channel_attention = self.fbs(x)
            self.cached_channel_attention = self.k_takes_all(self.cached_raw_channel_attention)
            
            channel_attention = self.cached_channel_attention
        
        raw_res = self.linear(x)
        
        return channel_attention.unsqueeze(1) * raw_res

class ElasticGPTUtil(ElasticDNNUtil):
    def convert_raw_dnn_to_master_dnn(self, raw_dnn: nn.Module, r: float, ignore_layers=[]):
        assert len(ignore_layers) == 0, 'not supported yet'

        raw_vit = deepcopy(raw_dnn)
        
        # set_module(module, 'patch_embed.proj', ProjConv_WrappedWithFBS(module.patch_embed.proj, r))
                
        for name, module in raw_vit.named_modules():
            # if name.endswith('attn'):
            #     set_module(module, 'qkv', ToQKV_WrappedWithFBS(module.qkv, r))
            if name.endswith('intermediate'):
                set_module(module, 'dense', Linear_WrappedWithFBS(module.dense, r))
            elif name.endswith('mlp'):
                set_module(module, 'c_fc', Linear_WrappedWithFBS(module.c_fc, r))


        return raw_vit
    
    def set_master_dnn_sparsity(self, master_dnn: nn.Module, sparsity: float):
        # for name, module in master_dnn.named_modules():
        #     if not name.endswith('attn'):
        #         continue
            
        #     q_features = module.qkv.to_qk.out_features // 2
            
        #     if (q_features - int(q_features * sparsity)) % module.num_heads != 0:
        #         # tune sparsity to ensure #unpruned channel % num_heads == 0
        #         # so that the pruning seems to reduce the dim_head of each head
        #         tuned_sparsity = 1. - int((q_features - int(q_features * sparsity)) / module.num_heads) * module.num_heads / q_features
        #         logger.debug(f'tune sparsity from {sparsity:.2f} to {tuned_sparsity}')
        #         sparsity = tuned_sparsity
        #         break
        
        return super().set_master_dnn_sparsity(master_dnn, sparsity)
    
    def select_most_rep_sample(self, master_dnn: nn.Module, samples: torch.Tensor):
        # print(samples)
        # return samples[0].unsqueeze(0)
        ret_di = samples['return_dict']
        del samples['return_dict']
        res = {k: v[0: 1] for k, v in samples.items()}
        res['return_dict'] = ret_di
        return res
        
    def extract_surrogate_dnn_via_samples(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):
        sample = self.select_most_rep_sample(master_dnn, samples)
        # assert sample.dim() == 4 and sample.size(0) == 1
        
        # print('before')
        master_dnn.eval()
        self.clear_cached_channel_attention_in_master_dnn(master_dnn)
        with torch.no_grad():
            master_dnn_output = master_dnn(**sample)
            
        # print('after')
        
        boosted_vit = deepcopy(master_dnn)
        
        def get_unpruned_indexes_from_channel_attn(channel_attn: torch.Tensor, k):
            assert channel_attn.size(0) == 1, 'use A representative sample to generate channel attentions'
            
            # print('attn_in_unpruned', channel_attn[0][0: 10])
            
            res = channel_attn[0].nonzero(as_tuple=True)[0] # should be one-dim

            # res = channel_attn[0].argsort(descending=True)[0: -int(channel_attn.size(1) * k)].sort()[0]
            
            # g = channel_attn
            # k = g.size(1) - int(g.size(1) * k)
            # res = g.topk(k, 1)[1][0].sort()[0]
            
            return res
        
        unpruned_indexes_of_layers = {}
        
        # for attn, ff in boosted_vit.transformer.layers:
        # for block_i, block in enumerate(boosted_vit.blocks):
        for block_i, block in enumerate(boosted_vit.model.transformer.h):
            # attn = block.attn
            # ff = block.mlp
            
            ff_0 = get_module(block, f'mlp.c_fc')
            # ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, k)
            ff_0_pruned_indexes = ff_0.k_takes_all.cached_i[0].sort()[0]
            ff_0_unpruned_indexes = torch.LongTensor([ii for ii in range(ff_0.cached_channel_attention.size(1)) if ii not in ff_0_pruned_indexes])
            new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
            new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
            if ff_0.linear.bias is not None:
                new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
            set_module(block, 'mlp.c_fc', nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
            
            ff_1 = get_module(block, f'mlp.c_proj')
            new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
            new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
            if ff_1.bias is not None:
                new_ff_1.bias.data.copy_(ff_1.bias.data)
            set_module(block, 'mlp.c_proj', new_ff_1)
            
            unpruned_indexes_of_layers[f'model.transformer.h.{block_i}.mlp.c_fc.0.weight'] = ff_0_unpruned_indexes
        
        surrogate_dnn = boosted_vit
        surrogate_dnn.eval()
        surrogate_dnn = surrogate_dnn.to(get_model_device(master_dnn))
        # logger.debug(surrogate_dnn)
        with torch.no_grad():
            surrogate_dnn_output = surrogate_dnn(**sample)
            
        output_diff = ((surrogate_dnn_output - master_dnn_output) ** 2).sum()
        # assert output_diff < 1e-4, output_diff
        logger.info(f'output diff of master and surrogate DNN: {output_diff}')
        logger.debug(f'example output of master/surrogate: {master_dnn_output.sum(0)[0: 10]}, {surrogate_dnn_output.sum(0)[0: 10]}')
        # logger.info(f'\nonly prune mlp!!!!\n')
        # logger.info(f'\nonly prune mlp!!!!\n')
        
        if return_detail:
            return boosted_vit, unpruned_indexes_of_layers
        
        return boosted_vit
    
    def extract_surrogate_dnn_via_samples_with_perf_test(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):
        master_dnn_size = get_model_size(master_dnn, True)
        master_dnn_latency = self._get_model_latency(master_dnn, samples, 50, 
                                               get_model_device(master_dnn), 50, False)
        
        res = self.extract_surrogate_dnn_via_samples(master_dnn, samples, return_detail)
        if not return_detail:
            surrogate_dnn = res
        else:
            surrogate_dnn, unpruned_indexes_of_layers = res
        surrogate_dnn_size = get_model_size(surrogate_dnn, True)
        surrogate_dnn_latency = self._get_model_latency(master_dnn, samples, 50, 
                                               get_model_device(master_dnn), 50, False)

        logger.info(f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample) -> '
                    f'surrogate DNN ({surrogate_dnn_size:.3f}MB, {surrogate_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(master_dnn_size / surrogate_dnn_size):.2f}x, '
                    f'latency: ↓ {(master_dnn_latency / surrogate_dnn_latency):.2f}x)')
        
        return res
    
    def _get_model_latency(self, model: torch.nn.Module, model_input_size, sample_num: int, 
                           device: str, warmup_sample_num: int, return_detail=False):
        import time
        
        if isinstance(model_input_size, tuple):
            dummy_input = torch.rand(model_input_size).to(device)
        else:
            dummy_input = model_input_size
            
        model = model.to(device)
        model.eval()
        
        # warm up
        with torch.no_grad():
            for _ in range(warmup_sample_num):
                model(**dummy_input)
                
        infer_time_list = []
                
        if device == 'cuda' or 'cuda' in str(device):
            with torch.no_grad():
                for _ in range(sample_num):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    model(**dummy_input)
                    e.record()
                    torch.cuda.synchronize()
                    cur_model_infer_time = s.elapsed_time(e) / 1000.
                    infer_time_list += [cur_model_infer_time]

        else:
            with torch.no_grad():
                for _ in range(sample_num):
                    start = time.time()
                    model(**dummy_input)
                    cur_model_infer_time = time.time() - start
                    infer_time_list += [cur_model_infer_time]
                    
        avg_infer_time = sum(infer_time_list) / sample_num

        if return_detail:
            return avg_infer_time, infer_time_list
        return avg_infer_time