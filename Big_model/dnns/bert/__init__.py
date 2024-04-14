from transformers import BertTokenizer, BertModel, BertConfig
from utils.dl.common.model import set_module
from torch import nn
import torch
from utils.common.log import logger


bert_model_tag = 'bert-base-multilingual-cased'


class BertForSenCls(nn.Module):
    def __init__(self, num_classes):
        super(BertForSenCls, self).__init__()
        
        logger.info(f'init bert for sen cls (using {bert_model_tag})')
        self.bert = BertModel.from_pretrained(bert_model_tag)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, **x):
        x['return_dict'] = False
        
        pool_output = self.bert(**x)[-1]
        
        return self.classifier(pool_output)
    
    
class BertForTokenCls(nn.Module):
    def __init__(self, num_classes):
        super(BertForTokenCls, self).__init__()
        
        logger.info(f'init bert for token cls (using {bert_model_tag})')
        self.bert = BertModel.from_pretrained(bert_model_tag)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, **x):
        x['return_dict'] = False
        
        pool_output = self.bert(**x)[0]
        
        return self.classifier(pool_output)


class BertForTranslation(nn.Module):
    def __init__(self):
        super(BertForTranslation, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_tag)
        
        vocab_size = BertConfig.from_pretrained(bert_model_tag).vocab_size
        self.decoder = nn.Linear(768, vocab_size)
        
        logger.info(f'init bert for sen cls (using {bert_model_tag}), vocab size {vocab_size}')
        
        # https://github.com/huggingface/transformers/blob/66954ea25e342fd451c26ec1c295da0b8692086b/src/transformers/models/bert_generation/modeling_bert_generation.py#L594
        self.decoder.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(self, **x):
        x['return_dict'] = False
        
        seq_output = self.bert(**x)[0]
        
        return self.decoder(seq_output)


def bert_base_sen_cls(num_classes):
    return BertForSenCls(num_classes)


def bert_base_token_cls(num_classes):
    return BertForTokenCls(num_classes)


def bert_base_translation(no_bert_pooler=False):
    # return BertForTranslation()
    from transformers import BertTokenizer, BertModel, BertConfig, EncoderDecoderModel, BertGenerationDecoder
    encoder = BertModel.from_pretrained(bert_model_tag)
    model = BertGenerationDecoder.from_pretrained(bert_model_tag)
    model.bert = encoder
    
    if no_bert_pooler:
        logger.info('replace pooler with nn.Identity()')
        encoder.pooler = nn.Identity()
    
    return model
    