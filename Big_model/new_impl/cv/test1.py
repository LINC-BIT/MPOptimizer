# from transformers import CvtModel,CvtConfig,CvtForImageClassification,AutoFeatureExtractor
# import torch
# from PIL import Image
# import requests
# from dnns.vit import vit_b_16
# torch.cuda.set_device(1)
# device = 'cuda'
# #configuration = CvtConfig(num_labels=5)
# # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# # image = Image.open(requests.get(url, stream=True).raw)
# # feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/cvt-13')
# model = CvtForImageClassification.from_pretrained("/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/cvt_model")#这里是规定最终我输出的分类个数，需要注意的是如果linear最终的输出不匹配的话，需要把第三个参数设置为True
# sample = torch.rand((4, 3, 224, 224)).to(device)
# model3 = vit_b_16(pretrained = True,num_classes=20)
# model2 = torch.load("/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/entry_model/cvt_pretrained.pt",map_location=device)
# model2['main'].train()
# for n, m in model2['main'].named_modules():
#     print(n)
#     if n=='cvt.encoder.stages.2.layers.2.attention.attention.convolution_projection_value.linear_projection':
#         print(m)
#     elif n== 'cvt.encoder.stages.2.layers.0.intermediate.dense':
#         print(m)
# outputs = model2['main'](sample)
# # print(**inputs)

import numpy 
a = [1,2,3,4,5]
print(a[1:])
