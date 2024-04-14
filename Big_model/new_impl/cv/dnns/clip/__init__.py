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

from transformers import CLIPProcessor, CLIPModel, CLIPVisionConfig, CLIPConfig
from dnns.clip.custom_clip import CLIPModelCanReceiveTextEmbeds

import torch.nn.functional as F


class Clip_ViTB16(nn.Module):
    def __init__(self, img_size):
        super(Clip_ViTB16, self).__init__()
        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model: CLIPModel = CLIPModelCanReceiveTextEmbeds.from_pretrained("openai/clip-vit-base-patch16")

        self.img_size = img_size
        
        # reconstruct xx
        vm_embed = self.model.vision_model.embeddings
        raw_num_patches = vm_embed.num_patches
        vm_embed.num_patches = (img_size // self.model.vision_model.embeddings.patch_size) ** 2
        vm_embed.num_positions = vm_embed.num_patches + 1
        vm_embed.register_buffer("position_ids", torch.arange(vm_embed.num_positions).expand((1, -1)), persistent=False)
        
        logger.info(f'due to changed input image size ({img_size}), num patches are updated from {raw_num_patches} to {vm_embed.num_patches}')

        self.first_inference = True
        
    def forward(self, images, texts: Union[List[List[str]], torch.Tensor], for_training, disable_return_loss=False, only_return_logits_per_text=False, no_grad_text=False):

        if isinstance(texts[0], str):
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        else:
            # input embeds instead of input ids
            # however, original CLIP cannot receive Tensor as input
            inputs = self.processor(images=images, return_tensors="pt")
            inputs['attention_mask'] = torch.ones((texts.size(0), texts.size(1)))
            inputs['input_embeds'] = texts
            
        if for_training and not disable_return_loss:
            inputs['return_loss'] = True
        else:
            inputs['return_loss'] = False
            
        inputs['only_return_logits_per_text'] = only_return_logits_per_text
        inputs['no_grad_text'] = no_grad_text
            
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to('cuda')
                
        if self.first_inference:
            logger.info(f'before input size: {inputs["pixel_values"].size()}')
        
        # print(inputs.keys())
        # print(inputs['pixel_values'].size())
        inputs['pixel_values'] = F.interpolate(inputs['pixel_values'], size=(self.img_size, self.img_size))
        # print(inputs['pixel_values'].size())
        
        if self.first_inference:
            logger.info(f'after input size: {inputs["pixel_values"].size()}')
            self.first_inference = False
        
        return self.model(**inputs)

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
def clip_vit_b_16(img_size):
    # https://huggingface.co/openai/clip-vit-base-patch16
    return Clip_ViTB16(img_size)
    
    


if __name__ == '__main__':
    model = clip_vit_b_16().cuda()
    # print(model)
    # exit()
    
    
    # config = CLIPConfig.from_pretrained('openai/clip-vit-base-patch16')
    # print(config)
    
    # # test 1: single image inference
    # from PIL import Image
    # import requests
    # image = Image.open('/data/zql/datasets/Caltech-256/data/caltech256/256_ObjectCategories/003.backpack/003_0001.jpg')
    # text = ["a photo of a dog", "a photo of a backpack", "a photo of a cat"]

    # o = model(image, text, False)
    # print(o)
    # print(o.logits_per_image.softmax(dim=1))
    
    # o = model(image, torch.load('dnns/clip/test_input_embed.pth'), False)
    # # print(o)
    # print(o.logits_per_image.softmax(dim=1))
    # exit()
    
    # test 2: normal training using clip loss (batch)
    from data import get_dataset, build_dataloader
    from torchvision.transforms import Compose, ToTensor, Resize
    dataset = get_dataset('Caltech256', '/data/zql/datasets/Caltech-256/data/caltech256/256_ObjectCategories/', 'train', transform=Compose([
        Resize((32, 32)), ToTensor()
    ]))
    dataloader = build_dataloader(dataset, 8, 0, True, None)

    from PIL import Image
    import requests
    images, labels = next(iter(dataloader))
    
    # torch.save(images, 'dnns/clip/test_image.pth')
    classes = dataset.classes
    text = [f"a photo of a {classes[i]}" for i in labels] # should be ground truth
    print(text)
    print(images.size())
    
    o = model(images, text, True)
    print(o)
    print(o.logits_per_image.softmax(dim=1))
    
    # o = model(image, torch.load('dnns/clip/test_input_embed.pth'), False)
    # # print(o)
    # print(o.logits_per_image.softmax(dim=1))