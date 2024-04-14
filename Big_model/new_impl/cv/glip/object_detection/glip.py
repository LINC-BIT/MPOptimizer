import timm
from timm.models._factory import load_checkpoint
import torch
import os
from typing import List, Union, Optional, Tuple
from torch import nn 
from torch.jit import Final
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.dl.common.model import get_model_device, set_module, get_module, get_model_latency, get_model_size, LayerActivation3
import torch.nn.functional as F
from utils.common.log import logger
from transformers import AutoTokenizer
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.detector.generalized_vl_rcnn import GeneralizedVLRCNN
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList
from torchvision import transforms as T
import matplotlib.pyplot as plt
import nltk
import re
from copy import deepcopy
from abc import ABC, abstractmethod
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util, LoRA
from new_impl.cv.elasticdnn.api.model import ElasticDNN_OfflineFMModel, ElasticDNN_OfflineMDModel
from methods.elasticdnn.model.base import Abs, KTakesAll, ElasticDNNUtil, Layer_WrappedWithFBS
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import BertConfig
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def collect_mm_fn(batch):
    if len(batch[0]) == 2:
        dict = {'images' : [], 'targets' : []}
    else:
        dict = {'images' : [], 'targets' : [], "info_imgs" : [], "ids" : []}

    for item in batch:
        if len(item) == 2:
            img, new_target = item
            if len(new_target) == 0:
                continue
            dict['images'].append(img)
            dict['targets'].append(new_target)
        else:
            img, new_target, info_imgs, ids = item
            if len(new_target) == 0:
                continue
            dict['images'].append(img)
            dict['targets'].append(new_target)
            dict['info_imgs'].append(info_imgs)
            dict['ids'].append(ids)

    return dict, torch.Tensor([0])

def run_ner(caption):
        noun_phrases = find_noun_phrases(caption)
        noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
        noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
        relevant_phrases = noun_phrases
        labels = noun_phrases

        tokens_positive = []

        for entity, label in zip(relevant_phrases, labels):
            try:
                # search all occurrences and mark them as different entities
                for m in re.finditer(entity, caption.lower()):
                    tokens_positive.append([[m.start(), m.end()]])
            except:
                print("noun entities:", noun_phrases)
                print("entity:", entity)
                print("caption:", caption.lower())

        return tokens_positive

def build_transform(cfg, min_image_size):
    """
    Creates a basic transformation that was used to train the models
    """

    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(min_image_size) if min_image_size is not None else lambda x: x,
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

def remove_punctuation(text: str) -> str:
    punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()

def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token

def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print("beg:", beg, "end:", end)
                print("token_positive:", tokens_positive)
                # print("beg_pos:", beg_pos, "end_pos:", end_pos)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

def find_noun_phrases(caption: str) -> List[str]:
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases

class Glip(nn.Module):
    def __init__(self, config, pretrain_path, min_image_size=None,confidence_threshold=0.7):
        super(Glip, self).__init__()
        state_dict = torch.load(pretrain_path)['model']
        self.min_image_size = min_image_size
        self.cfg = config
        self.confidence_threshold = confidence_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_PATH)
        self.device = torch.device(cfg.MODEL.DEVICE)
        for k in list(state_dict.keys()):
            if k.startswith('module'):
                new_k = k.replace('module.', '')
                state_dict[new_k] = state_dict.pop(k)
        self.model = GeneralizedVLRCNN(config)
        self.model.load_state_dict(state_dict, strict=False)
        # self.transform = build_transform(config, min_image_size)

    def forward(self, images, targets, for_training=None):
        # img_list = []
        # for image in images:
        #     img_list.append(self.transform(image).to(self.device))

        # if isinstance(texts, list):
        #     # we directly provided a list of category names
        #     caption_string = ""
        #     tokens_positive = []
        #     seperation_tokens = " . "
        #     for word in texts:
                
        #         tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
        #         caption_string += word
        #         caption_string += seperation_tokens
            
        #     tokenized = self.tokenizer([caption_string], return_tensors="pt")
        #     tokens_positive = [tokens_positive]

        #     texts = [caption_string]
        #     print(tokens_positive)
        # else:
        device = torch.device(cfg.MODEL.DEVICE)
        images = [image.to(device) for image in images]
        targets = [target.to(device) for target in targets]
        texts = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        positive_map = []
        # if custom_entity is None:
        # tokens_positive = self.run_ner(texts)
        # print(tokens_positive)
        # process positive map

        if self.training == False:
            try:
                tokens_positive = run_ner(texts[0])
            except:
                print('a')
            tokenized = self.tokenizer(texts, return_tensors="pt")
            positive_map = create_positive_map(tokenized, tokens_positive)
            if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
                plus = 1
            else:
                plus = 0
            positive_map = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        else:
            for i, text in enumerate(texts):
                tokenized = self.tokenizer(text, return_tensors="pt")
                tokens_positive = targets[i].get_field('tokens_positive')
                positive_map.append(create_positive_map(tokenized, tokens_positive))

            positive_map = torch.cat(positive_map, dim=0).to(device)


        if self.training:
            proposal_losses = self.model(images, targets, texts, positive_map=positive_map)
            return proposal_losses
        else:
            proposals, token_logits, dot_product_logits = self.model(images, targets, texts, positive_map=positive_map)
            proposal = self._post_process(proposals[0])
            return proposal, token_logits, dot_product_logits

    def _post_process_fixed_thresh(self, predictions):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = self.confidence_threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = self.confidence_threshold[0]
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def _post_process(self, predictions, threshold=0.5):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = threshold
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

# @torch.no_grad()
# def clip_vit_b_16():
#     # https://huggingface.co/openai/clip-vit-base-patch16
#     model = CLIPModelCanReceiveTextEmbeds.from_pretrained("openai/clip-vit-base-patch16")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
#     print(model)
    
#     from PIL import Image
#     import requests
#     image = Image.open('/data/zql/datasets/Caltech-256/data/caltech256/256_ObjectCategories/003.backpack/003_0001.jpg')
#     inputs = processor(text=["a photo of a dog", "a photo of a backpack", "a photo of a cat"], images=image, return_tensors="pt", padding=True)
#     print(inputs)
    
#     from utils.dl.common.model import LayerActivation2, get_module
#     input_embed_hook = LayerActivation2(get_module(model, 'text_model.embeddings'))
#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image # this is the image-text similarity score
#     probs = logits_per_image.softmax(dim=1)
#     print(probs)
    
#     input_embed = input_embed_hook.output
#     input_embed_hook.remove()
    
#     torch.save(input_embed, os.path.join(os.path.dirname(__file__), './test_input_embed.pth'))
    
#     print('embed', input_embed.size())
    
#     del inputs['input_ids']
#     inputs['input_embeds'] = input_embed
#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image # this is the image-text similarity score
#     probs = logits_per_image.softmax(dim=1)
#     print(probs)


@torch.no_grad()
def glip_model(config_path, pretrain_path):
    # https://huggingface.co/openai/clip-vit-base-patch16
    cfg.merge_from_file(config_path)
    return cfg, Glip(cfg, pretrain_path)

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

def get_model_latency_2(model: torch.nn.Module, sample: dict, sample_num: int, 
                      device: str, warmup_sample_num: int, return_detail=False):
    """Get the latency (inference time) of a PyTorch model.
    
    Reference: https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/

    Args:
        model (torch.nn.Module): A PyTorch model.
        model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.
        sample_num (int): How many inputs which size is :attr:`model_input_size` will be tested and compute the average latency as result.
        device (str): Typically be 'cpu' or 'cuda'.
        warmup_sample_num (int): Let model perform some dummy inference to warm up the test environment to avoid measurement loss.
        return_detail (bool, optional): Beside the average latency, return all result measured. Defaults to False.

    Returns:
        Union[float, Tuple[float, List[float]]]: The average latency (and all lantecy data) of :attr:`model`.
    """
    # if isinstance(model_input_size, tuple):
    #     dummy_input = torch.rand(model_input_size).to(device)
    # else:
    #     dummy_input = model_input_size
        
    model = model.to(device)
    model.eval()
    
    # warm up
    with torch.no_grad():
        for _ in range(warmup_sample_num):
            model(**sample)
            
    infer_time_list = []
            
    if device == 'cuda' or 'cuda' in str(device):
        with torch.no_grad():
            for _ in range(sample_num):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                model(**sample)
                e.record()
                torch.cuda.synchronize()
                cur_model_infer_time = s.elapsed_time(e) / 1000.
                infer_time_list += [cur_model_infer_time]

    else:
        with torch.no_grad():
            for _ in range(sample_num):
                start = time.time()
                model(**sample)
                cur_model_infer_time = time.time() - start
                infer_time_list += [cur_model_infer_time]
                
    avg_infer_time = sum(infer_time_list) / sample_num

    if return_detail:
        return avg_infer_time, infer_time_list
    return avg_infer_time

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D
        self.clamp_min_for_underflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW
        self.clamp_max_for_overflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

class BertSelfAttentionPrunable(BertSelfAttention):
    def __init__(self):
        config = BertConfig.from_pretrained('new_impl/cv/glip/object_detection/bert-base-uncased')
        super(BertSelfAttentionPrunable, self).__init__(config)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.query.out_features,) # NOTE: modified
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    @staticmethod
    def init_from_exist_self_attn(attn: BertSelfAttention):
        # print(attn)
        
        res = BertSelfAttentionPrunable()
        
        for attr in dir(attn):
            # if str(attr) in ['transpose_for_scores'] or str(attr).startswith('_'):
            #     continue
            # if isinstance(getattr(attn, attr), nn.Module):
                # print(attr)
                
            if isinstance(getattr(attn, attr), nn.Module):
                try:
                    # print(attr, 'ok')
                    setattr(res, attr, getattr(attn, attr))
                    
                except Exception as e:
                    print(attr, str(e))
        
        
        
        return res

class FM_to_MD_GLIP_Util(FM_to_MD_Util):
    def init_md_from_fm_by_reducing_width_with_perf_test_2(self, fm: nn.Module, reducing_width_ratio: int,
                                                         samples: torch.Tensor) -> nn.Module:
        fm_size = get_model_size(fm, True)
        fm_latency = get_model_latency_2(fm, samples, 20, 
                                               get_model_device(fm), 20, False)
        
        master_dnn = self.init_md_from_fm_by_reducing_width(fm, reducing_width_ratio)
        master_dnn_size = get_model_size(master_dnn, True)
        logger.debug(f'inited master DNN: {master_dnn}')
        # from utils.dl.common.model import get_module
        # print('after generating')
        # get_module(fm, 'head').debug()
        # get_module(master_dnn, 'head').debug()
        # print('test master latency')
        master_dnn_latency = get_model_latency_2(fm, samples, 20, 
                                               get_model_device(fm), 20, False)

        logger.info(f'init master DNN (w/o FBS yet) by reducing foundation model\'s width (by {reducing_width_ratio:d}x)')
        logger.info(f'foundation model ({fm_size:.3f}MB, {fm_latency:.4f}s/sample) -> '
                    f'master DNN ({master_dnn_size:.3f}MB, {master_dnn_latency:.4f}s/sample)\n'
                    f'(model size: ↓ {(fm_size / master_dnn_size):.2f}x, '
                    f'latency: ↓ {(fm_latency / master_dnn_latency):.2f}x)')
        
        return master_dnn

    def init_md_from_fm_by_reducing_width(self, fm: nn.Module, reducing_width_ratio: int, sparsity=0.0) -> nn.Module:
        #sparsity: it is mainly used to make a distilled model used in the baseline algorithm, and the parameter can ensure that the model has the same size as the model used in the online algorithm.
        fm_vit = deepcopy(fm)
        
        def _f(n):
            return int(n // reducing_width_ratio)
        
        # def _rand_indexes(n):
            # return torch.randperm(n)[0: int(n // reducing_width_ratio)]
            
        def l1_max_indexes(p: torch.Tensor, dim=0):
            assert dim in [0, 1]
            assert p.dim() in [1, 2, 4]
            
            if dim == 1:
                p = p.T
            
            p_norm = p.abs().contiguous().view(p.size(0), -1).sum(dim=1)
            n = p.size(0)

            t1 = p_norm.argsort(descending=True)[0: int(n // reducing_width_ratio)]
            t2 = t1.sort()[0]
            return p_norm.argsort(descending=True)[0: int(n // reducing_width_ratio)].sort()[0]
        
        def l1_max_indexes_with_sparsity(p: torch.Tensor, dim=0):
            assert dim in [0, 1]
            assert p.dim() in [1, 2, 4]
            
            if dim == 1:
                p = p.T
            
            p_norm = p.abs().contiguous().view(p.size(0), -1).sum(dim=1)
            n = p.size(0)
            return p_norm.argsort(descending=True)[0: int(n // reducing_width_ratio * (1 - sparsity))].sort()[0]
            
        for layer_i, layer in enumerate(fm_vit.model.backbone.body.layers):
            for block in layer.blocks:
                ori_attn = block.attn
                new_attn = WindowAttention(ori_attn.dim, ori_attn.window_size, ori_attn.num_heads, True, ori_attn.scale, 0., 0.)
                new_attn.relative_position_index = ori_attn.relative_position_index
                new_attn.relative_position_bias_table = ori_attn.relative_position_bias_table
                new_attn.qkv = ori_attn.qkv
                new_attn.attn_drop = ori_attn.attn_drop
                new_attn.proj = ori_attn.proj
                new_attn.proj_drop = ori_attn.proj_drop
                set_module(block, 'attn', new_attn)

        # first_attn = True
        for layer_i, layer in enumerate(fm_vit.model.backbone.body.layers):
            for block_i, block in enumerate(layer.blocks):
                qkv = block.attn.qkv
                new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                    qkv.bias is not None, qkv.weight.device)
                indexes = l1_max_indexes(qkv.weight.data, 0)
                new_qkv.weight.data.copy_(qkv.weight.data[indexes])
                if qkv.bias is not None:
                    new_qkv.bias.data.copy_(qkv.bias.data[indexes])
                # fm_vit.model.backbone.body.layers[0].blocks.0.attn.qkv
                set_module(fm_vit, f'model.backbone.body.layers.{layer_i}.blocks.{block_i}.attn.qkv', new_qkv)
                
                proj = block.attn.proj
                new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
                                    proj.bias is not None, proj.weight.device)
                new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
                if proj.bias is not None:
                    new_proj.bias.data.copy_(proj.bias.data)
                set_module(fm_vit, f'model.backbone.body.layers.{layer_i}.blocks.{block_i}.attn.proj', new_proj)
                
                fc1 = block.mlp.fc1
                new_fc1 = nn.Linear(fc1.in_features, int(_f(fc1.out_features) * (1 - sparsity)), 
                                    fc1.bias is not None, fc1.weight.device)
                indexes = l1_max_indexes_with_sparsity(fc1.weight.data, 0)
                new_fc1.weight.data.copy_(fc1.weight.data[indexes])
                if fc1.bias is not None:
                    new_fc1.bias.data.copy_(fc1.bias.data[indexes])
                set_module(fm_vit, f'model.backbone.body.layers.{layer_i}.blocks.{block_i}.mlp.fc1', new_fc1)

                fc2 = block.mlp.fc2
                new_fc2 = nn.Linear(int(_f(fc2.in_features) * (1 - sparsity)), fc2.out_features, 
                                    fc2.bias is not None, fc2.weight.device)
                new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes_with_sparsity(fc2.weight.data, 1)])
                if fc2.bias is not None:
                    new_fc2.bias.data.copy_(fc2.bias.data)
                set_module(fm_vit, f'model.backbone.body.layers.{layer_i}.blocks.{block_i}.mlp.fc2', new_fc2)

        for block in fm_vit.model.language_backbone.body.model.encoder.layer:
            set_module(block, 'attention.self', BertSelfAttentionPrunable.init_from_exist_self_attn(block.attention.self))

        for block_i, block in enumerate(fm_vit.model.language_backbone.body.model.encoder.layer):
            for k in ['query', 'key', 'value']:
                qkv = get_module(block, f'attention.self.{k}')
                new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                    qkv.bias is not None, qkv.weight.device)
                indexes = l1_max_indexes(qkv.weight.data, 0)
                new_qkv.weight.data.copy_(qkv.weight.data[indexes])
                if qkv.bias is not None:
                    new_qkv.bias.data.copy_(qkv.bias.data[indexes])
                set_module(block, f'attention.self.{k}', new_qkv)
            
            proj = get_module(block, f'attention.output.dense')
            new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
                                proj.bias is not None, proj.weight.device)
            new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
            if proj.bias is not None:
                new_proj.bias.data.copy_(proj.bias.data)
            set_module(block, f'attention.output.dense', new_proj)
            
            fc1 = get_module(block, f'intermediate.dense')
            new_fc1 = nn.Linear(fc1.in_features, int(_f(fc1.out_features) * (1 - sparsity)), 
                                fc1.bias is not None, fc1.weight.device)
            indexes = l1_max_indexes_with_sparsity(fc1.weight.data, 0)
            new_fc1.weight.data.copy_(fc1.weight.data[indexes])
            if fc1.bias is not None:
                new_fc1.bias.data.copy_(fc1.bias.data[indexes])
            set_module(block, f'intermediate.dense', new_fc1)

            fc2 = get_module(block, f'output.dense')
            new_fc2 = nn.Linear(int(_f(fc2.in_features) * (1 - sparsity)), fc2.out_features, 
                                fc2.bias is not None, fc2.weight.device)
            new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes_with_sparsity(fc2.weight.data, 1)])
            if fc2.bias is not None:
                new_fc2.bias.data.copy_(fc2.bias.data)
            set_module(block, f'output.dense', new_fc2)

        for block_i, block in enumerate(fm_vit.model.rpn.head.dyhead_tower):
            if block_i % 3 == 0:
                tmp = block.b_attn.attn
                tmp.head_dim = int(tmp.head_dim // reducing_width_ratio)
                tmp.embed_dim = int(tmp.embed_dim // reducing_width_ratio)
                set_module(block, 'b_attn.attn', tmp)
                for k in ['v_proj', 'l_proj', 'values_v_proj', 'values_l_proj']:
                    qkv = get_module(block, f'b_attn.attn.{k}')
                    new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                        qkv.bias is not None, qkv.weight.device)
                    indexes = l1_max_indexes(qkv.weight.data, 0)
                    new_qkv.weight.data.copy_(qkv.weight.data[indexes])
                    if qkv.bias is not None:
                        new_qkv.bias.data.copy_(qkv.bias.data[indexes])
                    set_module(block, f'b_attn.attn.{k}', new_qkv)

                for k in ['out_v_proj', 'out_l_proj']:
                    qkv = get_module(block, f'b_attn.attn.{k}')

                    new_qkv = nn.Linear(_f(qkv.in_features), qkv.out_features, 
                                        qkv.bias is not None, qkv.weight.device)
                    new_qkv.weight.data.copy_(qkv.weight.data[:, l1_max_indexes(qkv.weight.data, 1)])
                    if qkv.bias is not None:
                        new_qkv.bias.data.copy_(qkv.bias.data)
                    set_module(block, f'b_attn.attn.{k}', new_qkv)
                
            elif block_i % 3 == 1:
                set_module(block, 'attention.self', BertSelfAttentionPrunable.init_from_exist_self_attn(block.attention.self))
                for k in ['query', 'key', 'value']:
                    qkv = get_module(block, f'attention.self.{k}')

                    new_qkv = nn.Linear(qkv.in_features, _f(qkv.out_features), 
                                        qkv.bias is not None, qkv.weight.device)
                    indexes = l1_max_indexes(qkv.weight.data, 0)
                    
                    new_qkv.weight.data.copy_(qkv.weight.data[indexes])
                    if qkv.bias is not None:
                        new_qkv.bias.data.copy_(qkv.bias.data[indexes])
                    set_module(block, f'attention.self.{k}', new_qkv)
                
                proj = get_module(block, f'attention.output.dense')
                new_proj = nn.Linear(_f(proj.in_features), proj.out_features, 
                                    proj.bias is not None, proj.weight.device)
                new_proj.weight.data.copy_(proj.weight.data[:, l1_max_indexes(proj.weight.data, 1)])
                if proj.bias is not None:
                    new_proj.bias.data.copy_(proj.bias.data)
                set_module(block, f'attention.output.dense', new_proj)
                
                fc1 = get_module(block, f'intermediate.dense')
                new_fc1 = nn.Linear(fc1.in_features, int(_f(fc1.out_features) * (1 - sparsity)), 
                                    fc1.bias is not None, fc1.weight.device)
                indexes = l1_max_indexes_with_sparsity(fc1.weight.data, 0)
                new_fc1.weight.data.copy_(fc1.weight.data[indexes])
                if fc1.bias is not None:
                    new_fc1.bias.data.copy_(fc1.bias.data[indexes])
                set_module(block, f'intermediate.dense', new_fc1)

                fc2 = get_module(block, f'output.dense')
                new_fc2 = nn.Linear(int(_f(fc2.in_features) * (1 - sparsity)), fc2.out_features, 
                                    fc2.bias is not None, fc2.weight.device)
                new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes_with_sparsity(fc2.weight.data, 1)])
                if fc2.bias is not None:
                    new_fc2.bias.data.copy_(fc2.bias.data)
                set_module(block, f'output.dense', new_fc2)

            # reduce dim_embedding
            # if name.endswith('patch_embed.proj'):
            #     continue
                
            #     new_layer = nn.Conv2d(module.in_channels, _f(module.out_channels), module.kernel_size, module.stride,
            #                          module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode,
            #                          module.weight.device)
                
            #     rand_indexes = l1_max_indexes(module.weight.data)
            #     new_layer.weight.data.copy_(module.weight.data[rand_indexes])
            #     if new_layer.bias is not None:
            #         new_layer.bias.data.copy_(module.bias.data[rand_indexes])
                
            #     fm_vit.cls_token.data = fm_vit.cls_token.data[:, :, rand_indexes]
            #     fm_vit.pos_embed.data = fm_vit.pos_embed.data[:, :, rand_indexes]
            
            # elif isinstance(module, nn.Linear):
                
            #     if 'head' in name:
            #         continue
                
            #         new_layer = nn.Linear(_f(module.in_features), module.out_features, 
            #                             module.bias is not None, module.weight.device)
            #         new_layer.weight.data.copy_(module.weight.data[:, l1_max_indexes(module.weight.data, 1)])
            #         if new_layer.bias is not None:
            #             new_layer.bias.data.copy_(module.bias.data)
            #     else:
            #         first_attn = False
            #         if first_attn:
            #             first_attn = False
            #             new_layer = nn.Linear(module.in_features, _f(module.out_features), 
            #                                 module.bias is not None, module.weight.device)
                        
            #             rand_indexes = l1_max_indexes(module.weight.data)
            #             new_layer.weight.data.copy_(module.weight.data[rand_indexes])
            #             if new_layer.bias is not None:
            #                 new_layer.bias.data.copy_(module.bias.data[rand_indexes])
            #         else:
            #             new_layer = nn.Linear(_f(module.in_features), _f(module.out_features), 
            #                                 module.bias is not None, module.weight.device)
                        
            #             rand_indexes = l1_max_indexes(module.weight.data)
            #             new_layer.weight.data.copy_(module.weight.data[rand_indexes][:, l1_max_indexes(module.weight.data, 1)])
            #             if new_layer.bias is not None:
            #                 new_layer.bias.data.copy_(module.bias.data[rand_indexes])
                        
            # elif isinstance(module, nn.LayerNorm) and ('block' in name or name == 'norm' or name == 'norm.0'):
            #     new_layer = nn.LayerNorm(_f(module.normalized_shape[0]), eps=module.eps, device=module.weight.device)
            #     rand_indexes = l1_max_indexes(module.weight.data)
            #     new_layer.weight.data.copy_(module.weight.data[rand_indexes])
            #     new_layer.bias.data.copy_(module.bias.data[rand_indexes])
                
            # else:
            #     continue
            
            # original_layer_str = str(module)
            # set_module(fm_vit, name, new_layer)
            # logger.debug(f'set_module, {name}, {new_layer}')
            # logger.debug(f'slim {name} from {original_layer_str} to {new_layer}')
        
        return fm_vit

class FMLoRA_GLIP_Util(FMLoRA_Util):
    def train_only_lora_and_conv(self, fm: nn.Module):
        res = []
        for n, m in fm.named_modules():
            if isinstance(m, LoRA) or isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    p.requires_grad = True
                    res += [p]
            else:
                for p in m.parameters():
                    p.requires_grad = False
        return res


    @torch.no_grad()
    def add_lora_ab_to_fm(self, fm: nn.Module, ab_r: int, samples):
        fm.eval()
        
        # samples = {'images' : samples[0], 'targets' : samples[1]}

        for k, v in samples.items():
            if isinstance(v, torch.Tensor) or isinstance(v, BoxList):
                samples[k] = v.to(get_model_device(fm))
                print(k)

        _, o1_token_logits, o1_dot_product_logits = fm(**samples)
        
        mo_list = {k:v for k, v in fm.named_modules()}

        for name, module in fm.named_modules():
            if '.proj' in name or 'out' in name:
                continue
            if name.endswith(('k_proj', 'q_proj', 'v_proj', 'qkv', 'attn.proj', 'l_proj', 'query', 'key', 'value')):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
        
        _, o2_token_logits, o2_dot_product_logits = fm(**samples)
        
        output_diff = 0.
        for o1, o2 in list(zip(o1_dot_product_logits, o2_dot_product_logits)):
            output_diff += ((o1 - o2) ** 2).sum()

        if o1_token_logits is not None:
            output_diff += ((o1_token_logits - o2_token_logits) ** 2).sum()
        assert output_diff < 1e-5
        
        return fm
    
    @torch.no_grad()
    def absorb_lora_and_recover_net_structure(self, fm: nn.Module, samples: dict):       
        fm.eval()
        # print('absorb lora before')
        
        for k, v in samples.items():
            if isinstance(v, torch.Tensor):
                samples[k] = v.to(get_model_device(fm))
                print(k)

        _, o1_token_logits, o1_dot_product_logits = fm(**samples)
        
        for name, module in fm.named_modules():
            if not isinstance(module, ToQKV_WrappedWithLoRA):
                continue
            
            fc = module.fc
            ab = module.ab

            fc.weight.add_(ab[1].weight @ ab[0].weight)
            
            set_module(fm, name, fc)
        
        # print('absorb lora after')
        _, o2_token_logits, o2_dot_product_logits = fm(**samples)
        
        output_diff = 0.
        for o1, o2 in list(zip(o1_dot_product_logits, o2_dot_product_logits)):
            output_diff += ((o1 - o2) ** 2).sum()

        if o1_token_logits is not None:
            output_diff += ((o1_token_logits - o2_token_logits) ** 2).sum()
        assert output_diff < 1e-3, output_diff
        
        return fm

class ElasticDNN_OfflineMMDetFMModel(ElasticDNN_OfflineFMModel):
    def __init__(self, name: str, models_dict_path: str, device: str, num_classes=10, collate_fn=None):
        super().__init__(name, models_dict_path, device)
        self.num_classes = num_classes
        self.collate_fn = collate_fn
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        # print('DeeplabV3: start test acc')
        _d = test_loader.dataset
        from data import build_dataloader
        if _d.__class__.__name__ == 'MergedDataset':
            # print('\neval on merged datasets')
            datasets = _d.datasets
            if self.collate_fn is None:
                test_loaders = [build_dataloader(d, test_loader.batch_size, test_loader.num_workers, False, None, collate_fn=None) for d in datasets]
            else:
                test_loaders = [build_dataloader(d, test_loader.batch_size, test_loader.num_workers, False, None, collate_fn=self.collate_fn) for d in datasets]
            accs = [self.get_accuracy(loader) for loader in test_loaders]
            # print(accs)
            return sum(accs) / len(accs)
        
        # print('dataset len', len(test_loader.dataset))

        model = self.models_dict['main']
        device = self.device
        model.eval()

        # print('# classes', model.num_classes)
        
        model = model.to(device)
        from evaluator import COCOEvaluator, MMCOCODecoder
        from utils.common.others import HiddenPrints
        with torch.no_grad():
            with HiddenPrints():
                evaluator = COCOEvaluator(
                    dataloader=test_loader,
                    img_size=(416, 416),
                    confthre=0.01,
                    nmsthre=0.65,
                    num_classes=len(test_loader.dataset.classes),
                    testdev=True
                )
                res = evaluator.evaluate(model, False, False, decoder=MMCOCODecoder)
                map50 = res[1]
            # print('eval info', res[-1])
        return map50
    
    def infer(self, x, *args, **kwargs):
        if len(args) > 0:
            print(args, len(args))
            return self.models_dict['main'](x, *args) # forward(x, label)
        return self.models_dict['main'](**x)

class ElasticDNN_OfflineMMDetMDModel(ElasticDNN_OfflineMDModel):
    def __init__(self, name: str, models_dict_path: str, device: str, num_classes=10, collate_fn=None):
        super().__init__(name, models_dict_path, device)
        self.num_classes = num_classes
        self.collate_fn = collate_fn
        
    def get_accuracy(self, test_loader, *args, **kwargs):
        # print('DeeplabV3: start test acc')
        _d = test_loader.dataset
        from data import build_dataloader
        if _d.__class__.__name__ == 'MergedDataset':
            # print('\neval on merged datasets')
            datasets = _d.datasets
            if self.collate_fn is None:
                test_loaders = [build_dataloader(d, test_loader.batch_size, test_loader.num_workers, False, None, collate_fn=None) for d in datasets]
            else:
                test_loaders = [build_dataloader(d, test_loader.batch_size, test_loader.num_workers, False, None, collate_fn=self.collate_fn) for d in datasets]
            accs = [self.get_accuracy(loader) for loader in test_loaders]
            # print(accs)
            return sum(accs) / len(accs)
        
        # print('dataset len', len(test_loader.dataset))

        model = self.models_dict['main']
        device = self.device
        model.eval()

        # print('# classes', model.num_classes)
        
        model = model.to(device)
        from evaluator import COCOEvaluator, MMCOCODecoder
        from utils.common.others import HiddenPrints
        with torch.no_grad():
            with HiddenPrints():
                evaluator = COCOEvaluator(
                    dataloader=test_loader,
                    img_size=(416, 416),
                    confthre=0.01,
                    nmsthre=0.65,
                    num_classes=len(test_loader.dataset.classes),
                    testdev=True
                )
                res = evaluator.evaluate(model, False, False, decoder=MMCOCODecoder)
                map50 = res[1]
            # print('eval info', res[-1])
        return map50
    
    def infer(self, x, *args, **kwargs):
        if len(args) > 0:
            return self.models_dict['main'](x, *args) # forward(x, label)
        return self.models_dict['main'](**x)

class SqueezeLast(nn.Module):
    def __init__(self):
        super(SqueezeLast, self).__init__()
    
    def forward(self, x):
        return x.squeeze(-1)
    
    
class ProjConv_WrappedWithFBS(Layer_WrappedWithFBS):
    def __init__(self, raw_conv2d: nn.Conv2d, r):
        super(ProjConv_WrappedWithFBS, self).__init__()
        
        self.fbs = nn.Sequential(
            Abs(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(raw_conv2d.in_channels, raw_conv2d.out_channels // r),
            nn.ReLU(),
            nn.Linear(raw_conv2d.out_channels // r, raw_conv2d.out_channels),
            nn.ReLU()
        )
        
        self.raw_conv2d = raw_conv2d
        # self.raw_bn = raw_bn # remember clear the original BNs in the network
        
        nn.init.constant_(self.fbs[5].bias, 1.)
        nn.init.kaiming_normal_(self.fbs[5].weight)

    def forward(self, x):
        raw_x = self.raw_conv2d(x)
        
        if self.use_cached_channel_attention and self.cached_channel_attention is not None:
            channel_attention = self.cached_channel_attention
        else:
            self.cached_raw_channel_attention = self.fbs(x)
            self.cached_channel_attention = self.k_takes_all(self.cached_raw_channel_attention)
            
            channel_attention = self.cached_channel_attention
        
        return raw_x * channel_attention.unsqueeze(2).unsqueeze(3)


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
    
    
class ToQKV_WrappedWithFBS(Layer_WrappedWithFBS):
    """
    This regards to_q/to_k/to_v as a whole (in fact it consists of multiple heads) and prunes it.
    It seems different channels of different heads are pruned according to the input. 
    This is different from "removing some head" or "removing the same channels in each head".
    """
    def __init__(self, to_qkv: nn.Linear, r):
        super(ToQKV_WrappedWithFBS, self).__init__()
        
        # self.to_qkv = to_qkv
        
        self.to_qk = nn.Linear(to_qkv.in_features, to_qkv.out_features // 3 * 2, bias=to_qkv.bias is not None)
        self.to_v = nn.Linear(to_qkv.in_features, to_qkv.out_features // 3, bias=to_qkv.bias is not None)
        self.to_qk.weight.data.copy_(to_qkv.weight.data[0: to_qkv.out_features // 3 * 2])
        if to_qkv.bias is not None:
            self.to_qk.bias.data.copy_(to_qkv.bias.data[0: to_qkv.out_features // 3 * 2])
        self.to_v.weight.data.copy_(to_qkv.weight.data[to_qkv.out_features // 3 * 2: ])
        if to_qkv.bias is not None:
            self.to_v.bias.data.copy_(to_qkv.bias.data[to_qkv.out_features // 3 * 2: ])
                
        self.fbs = nn.Sequential(
            Rearrange('b n d -> b d n'),
            Abs(),
            nn.AdaptiveAvgPool1d(1),
            SqueezeLast(),
            nn.Linear(to_qkv.in_features, to_qkv.out_features // 3 // r),
            nn.ReLU(),
            # nn.Linear(to_qkv.out_features // 3 // r, to_qkv.out_features // 3),
            nn.Linear(to_qkv.out_features // 3 // r, self.to_v.out_features),
            nn.ReLU()
        )
        
        nn.init.constant_(self.fbs[6].bias, 1.)
        nn.init.kaiming_normal_(self.fbs[6].weight)
    
    def forward(self, x):
        if self.use_cached_channel_attention and self.cached_channel_attention is not None:
            channel_attention = self.cached_channel_attention
        else:
            self.cached_raw_channel_attention = self.fbs(x)
            
            # print()
            # for attn in self.cached_raw_channel_attention.chunk(3, dim=1)[0: 1]:
            #     print(self.cached_raw_channel_attention.size(), attn.size())
            #     print(self.k_takes_all.k)
            #     print(attn[0].nonzero(as_tuple=True)[0].size(), attn[0])
                
            self.cached_channel_attention = self.k_takes_all(self.cached_raw_channel_attention)
            
            
            # for attn in self.cached_channel_attention.chunk(3, dim=1)[0: 1]:
            #     print(self.cached_channel_attention.size(), attn.size())
            #     print(self.k_takes_all.k)
            #     print(attn[0].nonzero(as_tuple=True)[0].size(), attn[0])
            # print()
            
            channel_attention = self.cached_channel_attention
        
        qk = self.to_qk(x)
        v = channel_attention.unsqueeze(1) * self.to_v(x)
        return torch.cat([qk, v], dim=-1)
        
        # qkv = raw_res.chunk(3, dim = -1)
        
        # raw_v = qkv[2]
        # print('raw_k, raw_v', qkv[0].sum((0, 1))[0: 10], qkv[0].sum((0, 1)).nonzero(as_tuple=True)[0].size(),
        #       qkv[1].sum((0, 1))[0: 10], qkv[1].sum((0, 1)).nonzero(as_tuple=True)[0].size(),)
        # print('raw_v', raw_v.size(), raw_v.sum((0, 1))[0: 10], raw_v.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        
        # qkv_attn = channel_attention.chunk(3, dim=-1)
        # print('attn', [attn[0][0: 10] for attn in qkv_attn])
        # print(channel_attention.unsqueeze(1).size(), raw_res.size())
        # print('fbs', channel_attention.size(), raw_res.size())
        # return channel_attention.unsqueeze(1) * raw_res
    
    
class StaticFBS(nn.Module):
    def __init__(self, static_channel_attention):
        super(StaticFBS, self).__init__()
        assert static_channel_attention.dim() == 2 and static_channel_attention.size(0) == 1
        self.static_channel_attention = nn.Parameter(static_channel_attention, requires_grad=False) # (1, dim)
        
    def forward(self, x):
        # print('staticfbs', x, self.static_channel_attention.unsqueeze(1))
        return x * self.static_channel_attention.unsqueeze(1)

class ElasticGLIPUtil(ElasticDNNUtil):
    def convert_raw_dnn_to_master_dnn(self, raw_dnn: nn.Module, r: float, ignore_layers=[]):
        assert len(ignore_layers) == 0, 'not supported yet'

        raw_vit = deepcopy(raw_dnn)
        
        
                
        for name, module in raw_vit.named_modules():
            # if name.endswith('patch_embed'):
            #     set_module(module, 'proj', ProjConv_WrappedWithFBS(module.proj, r))
            # if name.endswith('attn') and not name.endswith('b_attn.attn') and not name.endswith('b_attn'):
            #     set_module(module, 'qkv', ToQKV_WrappedWithFBS(module.qkv, r))
            if name.endswith('intermediate'):
                set_module(module, 'dense', Linear_WrappedWithFBS(module.dense, r))
            elif name.endswith('mlp'):
                set_module(module, 'fc1', Linear_WrappedWithFBS(module.fc1, r))
        
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
        sample={}
        sample['images'] = [samples['images'][0]]
        sample['targets'] = [samples['targets'][0]]
        # return samples[0].unsqueeze(0)
        # res = {k: v[0: 1] for k, v in samples.items()}
        return sample
        
    def extract_surrogate_dnn_via_samples(self, master_dnn: nn.Module, samples: torch.Tensor, return_detail=False):#产生小模型的步骤
        sample = self.select_most_rep_sample(master_dnn, samples)
        # assert sample.dim() == 4 and sample.size(0) == 1
        
        # print('before')
        master_dnn.eval()
        self.clear_cached_channel_attention_in_master_dnn(master_dnn)
        with torch.no_grad():
            _, o1_token_logits, o1_dot_product_logits  = master_dnn(**sample)
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
        for layer_i, layer in enumerate(boosted_vit.model.backbone.body.layers):
            for block_i, block in enumerate(layer.blocks):
                # attn = block.attn
                # ff = block.mlp
                
                ff_0 = get_module(block, f'mlp.fc1')
                # ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, k)
                ff_0_pruned_indexes = ff_0.k_takes_all.cached_i[0].sort()[0]
                ff_0_unpruned_indexes = torch.LongTensor([ii for ii in range(ff_0.cached_channel_attention.size(1)) if ii not in ff_0_pruned_indexes])
                new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
                new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
                if ff_0.linear.bias is not None:
                    new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
                set_module(block, 'mlp.fc1', nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
                
                ff_1 = get_module(block, f'mlp.fc2')
                new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
                new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
                if ff_1.bias is not None:
                    new_ff_1.bias.data.copy_(ff_1.bias.data)
                set_module(block, 'mlp.fc2', new_ff_1)
                
                unpruned_indexes_of_layers[f'model.backbone.body.layers.{layer_i}.blocks.{block_i}.mlp.fc1.0.weight'] = ff_0_unpruned_indexes
        # for block_i,block in enumerate(boosted_vit.vision_model.encoder.layers):

        #     attn = block.self_attn
        #     ff = block.mlp
        #     ff_0 = ff.fc1
        #     # ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, k)
        #     ff_0_pruned_indexes = ff_0.k_takes_all.cached_i[0].sort()[0]
        #     ff_0_unpruned_indexes = torch.LongTensor([ii for ii in range(ff_0.cached_channel_attention.size(1)) if ii not in ff_0_pruned_indexes])
        #     new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
        #     new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
        #     if ff_0.linear.bias is not None:
        #         new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
        #     set_module(ff, 'fc1', nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
            
        #     ff_1 = ff.fc2
        #     new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
        #     new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
        #     if ff_1.bias is not None:
        #         new_ff_1.bias.data.copy_(ff_1.bias.data)
        #     set_module(ff, 'fc2', new_ff_1)
            
        #     unpruned_indexes_of_layers[f'vision_model.encoder.layers.{block_i}.mlp.fc1.0.weight'] = ff_0_unpruned_indexes


        # for block_i, block in enumerate(boosted_vit.text_decoder.bert.encoder.layer):
        #     # attn = block.attn
        #     # ff = block.mlp
            
        #     ff_0 = get_module(block, f'intermediate.dense')
        #     # ff_0_unpruned_indexes = get_unpruned_indexes_from_channel_attn(ff_0.cached_channel_attention, k)
        #     ff_0_pruned_indexes = ff_0.k_takes_all.cached_i[0].sort()[0]
        #     ff_0_unpruned_indexes = torch.LongTensor([ii for ii in range(ff_0.cached_channel_attention.size(1)) if ii not in ff_0_pruned_indexes])
        #     new_ff_0 = nn.Linear(ff_0.linear.in_features, ff_0_unpruned_indexes.size(0), ff_0.linear.bias is not None)
        #     new_ff_0.weight.data.copy_(ff_0.linear.weight.data[ff_0_unpruned_indexes])
        #     if ff_0.linear.bias is not None:
        #         new_ff_0.bias.data.copy_(ff_0.linear.bias.data[ff_0_unpruned_indexes])
        #     set_module(block, 'intermediate.dense', nn.Sequential(new_ff_0, StaticFBS(ff_0.cached_channel_attention[:, ff_0_unpruned_indexes])))
            
        #     ff_1 = get_module(block, f'output.dense')
        #     new_ff_1 = nn.Linear(ff_0_unpruned_indexes.size(0), ff_1.out_features, ff_1.bias is not None)
        #     new_ff_1.weight.data.copy_(ff_1.weight.data[:, ff_0_unpruned_indexes])
        #     if ff_1.bias is not None:
        #         new_ff_1.bias.data.copy_(ff_1.bias.data)
        #     set_module(block, 'output.dense', new_ff_1)
            
        #     unpruned_indexes_of_layers[f'text_decoder.bert.encoder.layer.{block_i}.intermediate.dense.0.weight'] = ff_0_unpruned_indexes
        surrogate_dnn = boosted_vit
        surrogate_dnn.eval()
        surrogate_dnn = surrogate_dnn.to(get_model_device(master_dnn))
        # logger.debug(surrogate_dnn)
        with torch.no_grad():
            _, o2_token_logits, o2_dot_product_logits  = surrogate_dnn(**sample)
            
        output_diff = 0.
        for o1, o2 in list(zip(o1_dot_product_logits, o2_dot_product_logits)):
            output_diff += ((o1 - o2) ** 2).sum()

        if o1_token_logits is not None:
            output_diff += ((o1_token_logits - o2_token_logits) ** 2).sum()
        # assert output_diff < 1e-4, output_diff
        logger.info(f'output diff of master and surrogate DNN: {output_diff}')
        # logger.debug(f'example output of master/surrogate: {master_dnn_output.sum(0)[0: 10]}, {surrogate_dnn_output.sum(0)[0: 10]}')
        # logger.info(f'\nonly prune mlp!!!!\n')
        # logger.info(f'\nonly prune mlp!!!!\n')
        
        if return_detail:
            return boosted_vit, unpruned_indexes_of_layers
        
        return boosted_vit
    
    def extract_surrogate_dnn_via_samples_with_perf_test(self, master_dnn: nn.Module, samples, return_detail=False):
        master_dnn_size = get_model_size(master_dnn, True)
        sample = {}
        sample['images'] = [samples['images'][0]]
        sample['targets'] = [samples['targets'][0]]
        master_dnn_latency = self._get_model_latency(master_dnn, sample, 50, 
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
    
    def _get_model_latency(self, model: torch.nn.Module, sample, sample_num: int, 
                           device: str, warmup_sample_num: int, return_detail=False):
        import time
            
        model = model.to(device)
        model.eval()
        sample['images'] = [sample['images'][0]]
        sample['targets'] = [sample['targets'][0]]
        # warm up
        with torch.no_grad():
            for _ in range(warmup_sample_num):
                model(**sample)
                
        infer_time_list = []
                
        if device == 'cuda' or 'cuda' in str(device):
            with torch.no_grad():
                for _ in range(sample_num):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    model(**sample)
                    e.record()
                    torch.cuda.synchronize()
                    cur_model_infer_time = s.elapsed_time(e) / 1000.
                    infer_time_list += [cur_model_infer_time]

        else:
            with torch.no_grad():
                for _ in range(sample_num):
                    start = time.time()
                    model(**sample)
                    cur_model_infer_time = time.time() - start
                    infer_time_list += [cur_model_infer_time]
                    
        avg_infer_time = sum(infer_time_list) / sample_num

        if return_detail:
            return avg_infer_time, infer_time_list
        return avg_infer_time


# from typing import Any, Dict
# from schema import Schema, Or
# import schema
# from data import Scenario, MergedDataset
# from methods.base.alg import BaseAlg
# from data import build_dataloader
# from ..model import ElasticDNN_OfflineFMModel, ElasticDNN_OfflineMDModel
# from ...model.base import ElasticDNNUtil
# import torch.optim
# import tqdm
# import torch.nn.functional as F
# from torch import nn
# from utils.dl.common.env import create_tbwriter
# import os
# import random
# import numpy as np
# from copy import deepcopy
# from utils.dl.common.model import LayerActivation2, get_module
# from utils.common.log import logger


# class ElasticDNN_Det_MDPretrainingWoFBSAlg(BaseAlg):
#     """
#     TODO: fine-tuned FM -> init MD -> trained MD -> construct indexes (only between similar weights) and fine-tune
#     """
#     def get_required_models_schema(self) -> Schema:
#         return Schema({
#             'fm': ElasticDNN_OfflineFMModel,
#             'md': ElasticDNN_OfflineMDModel
#         })
        
#     def get_required_hyp_schema(self) -> Schema:
#         return Schema({
#             'launch_tbboard': bool,
            
#             'samples_size': any,
#             'generate_md_width_ratio': int,
            
#             'train_batch_size': int,
#             'val_batch_size': int,
#             'num_workers': int,            
#             'optimizer': str,
#             'optimizer_args': dict,
#             'scheduler': str,
#             'scheduler_args': dict,
#             'num_iters': int,
#             'val_freq': int,
#             'distill_loss_weight': float
#         })

#     def run(self, scenario: Scenario, hyps: Dict) -> Dict[str, Any]:
#         super().run(scenario, hyps)
        
#         assert isinstance(self.models['md'], ElasticDNN_OfflineMDModel) # for auto completion
#         assert isinstance(self.models['fm'], ElasticDNN_OfflineFMModel) # for auto completion
        
#         # 1. add FBS
#         device = self.models['md'].device
        
#         if self.models['md'].models_dict['main'] == -1:
#             logger.info(f'init master DNN by reducing width of an adapted foundation model (already tuned by LoRA)...')
            
#             before_fm_model = deepcopy(self.models['fm'].models_dict['main'])
#             lora_util = self.models['fm'].get_lora_util()
            
#             sample = hyps['samples_size']
#             if isinstance(sample, (tuple, list)) and isinstance(sample[0], int):
#                 sample = torch.rand(hyps['samples_size']).to(device)
            
#             lora_absorbed_fm_model = lora_util.absorb_lora_and_recover_net_structure(self.models['fm'].models_dict['main'], 
#                                                                                     sample)
#             self.models['fm'].models_dict['main'] = lora_absorbed_fm_model
#             master_dnn = self.models['fm'].generate_md_by_reducing_width(hyps['generate_md_width_ratio'], 
#                                                                         sample)
#             self.models['fm'].models_dict['main'] = before_fm_model
            
#             self.models['md'].models_dict['main'] = master_dnn
#             self.models['md'].to(device)
        
#         # 2. train (knowledge distillation, index relationship)
#         offline_datasets = scenario.get_offline_datasets()
#         train_dataset = MergedDataset([d['train'] for d in offline_datasets.values()])
#         val_dataset = MergedDataset([d['val'] for d in offline_datasets.values()])
#         train_loader = iter(build_dataloader(train_dataset, hyps['train_batch_size'], hyps['num_workers'],
#                                         True, None))
#         val_loader = build_dataloader(val_dataset, hyps['val_batch_size'], hyps['num_workers'],
#                                       False, False)
        
#         # logger.info(f'FM acc: {self.models["fm"].get_accuracy(val_loader):.4f}')
        
#         # 2.1 train whole master DNN (knowledge distillation)
#         for p in master_dnn.parameters():
#             p.requires_grad = True
#         self.models['md'].to_train_mode()
        
#         optimizer = torch.optim.__dict__[hyps['optimizer']]([
#             {'params': self.models['md'].models_dict['main'].parameters(), **hyps['optimizer_args']}
#         ])
#         scheduler = torch.optim.lr_scheduler.__dict__[hyps['scheduler']](optimizer, **hyps['scheduler_args'])
#         tb_writer = create_tbwriter(os.path.join(self.res_save_dir, 'tb_log'), launch_tbboard=hyps['launch_tbboard'])
#         pbar = tqdm.tqdm(range(hyps['num_iters']), dynamic_ncols=True)
#         best_avg_val_acc = 0.
        
#         md_output_hook = None
        
#         for iter_index in pbar:
#             self.models['md'].to_train_mode()
#             self.models['fm'].to_eval_mode()
            
#             # rand_sparsity = random.random() * (hyps['max_sparsity'] - hyps['min_sparsity']) + hyps['min_sparsity']
#             # elastic_dnn_util.set_master_dnn_sparsity(self.models['md'].models_dict['main'], rand_sparsity)
#             if md_output_hook is None:
#                 md_output_hook = self.models['md'].get_feature_hook()
#                 fm_output_hook = self.models['fm'].get_feature_hook()
                
#             x, y = next(train_loader)
#             if isinstance(x, dict):
#                 for k, v in x.items():
#                     if isinstance(v, torch.Tensor):
#                         x[k] = v.to(device)
#                 y = y.to(device)
#             else:
#                 x, y = x.to(device), y.to(device)
            
#             with torch.no_grad():
#                 fm_output = self.models['fm'].infer(x)
#             task_loss = self.models['md'].forward_to_get_task_loss(x, y)
            
#             md_output = md_output_hook.output
#             fm_output = fm_output_hook.output
            
#             distill_loss = hyps['distill_loss_weight'] * self.models['md'].get_distill_loss(md_output, fm_output)
#             total_loss = task_loss + distill_loss
            
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#             scheduler.step()
            
#             if (iter_index + 1) % hyps['val_freq'] == 0:
                
#                 # elastic_dnn_util.clear_cached_channel_attention_in_master_dnn(self.models['md'].models_dict['main'])
#                 md_output_hook.remove()
#                 md_output_hook = None
#                 fm_output_hook.remove()
#                 fm_output_hook = None
                
#                 cur_md = self.models['md'].models_dict['main']
#                 md_for_test = deepcopy(self.models['md'].models_dict['main'])
#                 val_acc = 0.
                
#                 self.models['md'].models_dict['main'] = md_for_test
#                 self.models['md'].to_eval_mode()
#                 val_acc = self.models['md'].get_accuracy(val_loader)
                
#                 self.models['md'].models_dict['main'] = cur_md
                
#                 self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_last.pt'))
#                 self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_last.pt'))
                
#                 if val_acc > best_avg_val_acc:
#                     best_avg_val_acc = val_acc
#                     self.models['md'].save_model(os.path.join(self.res_save_dir, 'models/md_best.pt'))
#                     self.models['fm'].save_model(os.path.join(self.res_save_dir, 'models/fm_best.pt'))
                
#             tb_writer.add_scalars(f'losses', dict(task=task_loss, distill=distill_loss, total=total_loss), iter_index)
#             pbar.set_description(f'loss: {total_loss:.6f}')
#             if (iter_index + 1) >= hyps['val_freq']:
#                 tb_writer.add_scalar(f'accs/val_acc', val_acc, iter_index)
#                 pbar.set_description(f'loss: {total_loss:.6f}, val_acc: {val_acc:.4f}')

# if __name__ == '__main__':
#     model = glip_model('new_impl/cv/glip/object_detection/pretrained_model/glip_Swin_T_O365_GoldG.yaml','new_impl/cv/glip/object_detection/pretrained_model/glip_tiny_model_o365_goldg_cc_sbu.pth').cuda()
#     model.eval()
#     # print(model)
#     # exit()
    
    
#     # config = CLIPConfig.from_pretrained('openai/clip-vit-base-patch16')
#     # print(config)
    
#     # # test 1: single image inference
    # from PIL import Image, ImageDraw
    # import requests
    # import numpy as np
    # ori_image = Image.open('new_impl/cv/glip/object_detection/9472793441_b7822c00de_z.jpg').convert("RGB")
    # image = [np.asarray(ori_image)[:, :, [2, 1, 0]]]
    # text = 'sofa . remote . dog . person . car . sky . plane .'
    # target = torch.Tensor()
    # o = model(image, text)
    # o = model._post_process(o[0])
    # print(o)
    # bboxes = o.bbox.cpu()
    # a = ImageDraw.ImageDraw(ori_image)
    # for box in bboxes:
    #     box = box.int()
    #     a.rectangle(((box[0], box[1]), (box[2], box[3])), fill=None, outline='red', width=2)
    # ori_image.save('test.jpg')
#     # print(o.logits_per_image.softmax(dim=1))
    
#     # o = model(image, torch.load('dnns/clip/test_input_embed.pth'), False)
#     # # print(o)
#     # print(o.logits_per_image.softmax(dim=1))
#     # exit()
    
#     # test 2: normal training using clip loss (batch)
#     from data import get_dataset, build_dataloader
#     from torchvision.transforms import Compose, ToTensor, Resize
#     dataset = get_dataset('Caltech256', '/data/zql/datasets/Caltech-256/data/caltech256/256_ObjectCategories/', 'train', transform=Compose([
#         Resize((32, 32)), ToTensor()
#     ]))
#     dataloader = build_dataloader(dataset, 8, 0, True, None)

#     from PIL import Image
#     import requests
#     images, labels = next(iter(dataloader))
    
#     # torch.save(images, 'dnns/clip/test_image.pth')
#     classes = dataset.classes
#     text = [f"a photo of a {classes[i]}" for i in labels] # should be ground truth
#     print(text)
#     print(images.size())
    
#     o = model(images, text, True)
#     print(o)
#     print(o.logits_per_image.softmax(dim=1))
    
#     # o = model(image, torch.load('dnns/clip/test_input_embed.pth'), False)
#     # # print(o)
#     # print(o.logits_per_image.softmax(dim=1))