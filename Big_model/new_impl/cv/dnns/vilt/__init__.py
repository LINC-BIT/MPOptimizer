import timm
from timm.models._factory import load_checkpoint
import torch
import os
from typing import List, Union
from torch import nn 
from torch.jit import Final
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.dl.common.model import get_model_device, set_module
import torch.nn.functional as F
from utils.common.log import logger

from transformers import ViltModel, ViltForQuestionAnswering
import torch.nn.functional as F



def vilt_b_32(num_classes):
    """
    Vilt for VQA
    
    settings based on the dataset VQAv2 (3129 classes): 
    
    1. use half of classes for LoRA adaptation
    2. use this half of classes for DA evaluation (using corruptions for generating domain shifts), 
       and use another half of classes for CL evaluation.
    """
    
    model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-mlm-itm')

    linear = model.classifier[3]
    new_linear = nn.Linear(linear.in_features, num_classes, bias=True)
    set_module(model, 'classifier.3', new_linear)

    return model


if __name__ == '__main__':
    model = vilt_b_32(1565)
    
    print(model)
    
    from transformers import ViltProcessor, ViltModel
    from PIL import Image
    import requests

    # prepare image and text
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = "hello world"

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm-itm")

    inputs = processor(image, text, return_tensors="pt")
    
    print(inputs)
    
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    
    print(last_hidden_states.shape)