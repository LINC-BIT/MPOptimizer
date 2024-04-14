# import os
# import torch
# import pickle

# from .raw_vit import ViT


# def vit_b_16(pretrained_backbone=True):
#     vit = ViT(
#         image_size = 224,
#         patch_size = 16,
#         num_classes = 1000,
#         dim = 768, # encoder layer/attention input/output size (Hidden Size D in the paper)
#         depth = 12,
#         heads = 12, # (Heads in the paper)
#         dim_head = 64, # attention hidden size (seems be default, never change this)
#         mlp_dim = 3072, # mlp layer hidden size (MLP size in the paper)
#         dropout = 0.,
#         emb_dropout = 0.
#     )
    
#     if pretrained_backbone:
#         ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'weights/base_p16_224_backbone.pth'))
#         vit.load_state_dict(ckpt)
#     return vit
    
    
# def vit_l_16(pretrained_backbone=True):
#     vit =  ViT(
#         image_size = 224,
#         patch_size = 16,
#         num_classes = 1000,
#         dim = 1024, # encoder layer/attention input/output size (Hidden Size D in the paper)
#         depth = 24,
#         heads = 16, # (Heads in the paper)
#         dim_head = 64, # attention hidden size (seems be default, never change this)
#         mlp_dim = 4096, # mlp layer hidden size (MLP size in the paper)
#         dropout = 0.,
#         emb_dropout = 0.
#     )
    
#     if pretrained_backbone:
#         # https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k
#         ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'weights/pytorch_model.bin'))
#             # ckpt = pickle.load(f)
#         # print(ckpt)
#         # exit()
#         # ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'weights/large_p16_224_backbone.pth'))
#         vit.load_state_dict(ckpt)
#         # pass
#     return vit
    
    
    
# def vit_h_16():
#     return ViT(
#         image_size = 224,
#         patch_size = 16,
#         num_classes = 1000,
#         dim = 1280, # encoder layer/attention input/output size (Hidden Size D in the paper)
#         depth = 32,
#         heads = 16, # (Heads in the paper)
#         dim_head = 64, # attention hidden size (seems be default, never change this)
#         mlp_dim = 5120, # mlp layer hidden size (MLP size in the paper)
#         dropout = 0.,
#         emb_dropout = 0.
#     )

