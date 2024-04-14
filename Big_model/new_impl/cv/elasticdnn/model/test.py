import torch
from torch import nn

from methods.elasticdnn.model.base import ElasticDNNUtil


def test(raw_dnn: nn.Module, ignore_layers, elastic_dnn_util: ElasticDNNUtil, input_sample: torch.Tensor, sparsity):
    
    # raw_dnn.eval()
    # with torch.no_grad():
    #     raw_dnn(input_sample)
        
    master_dnn = elastic_dnn_util.convert_raw_dnn_to_master_dnn_with_perf_test(raw_dnn, 16, ignore_layers)
    # print(master_dnn)
    # exit()
    
    elastic_dnn_util.set_master_dnn_sparsity(master_dnn, sparsity)
    
    # master_dnn.eval()
    # with torch.no_grad():
    #     master_dnn(input_sample)
    
    surrogate_dnn = elastic_dnn_util.extract_surrogate_dnn_via_samples_with_perf_test(master_dnn, input_sample)
    
    
if __name__ == '__main__':
    from utils.dl.common.env import set_random_seed
    set_random_seed(1)
    
    # from torchvision.models import resnet50
    # from methods.elasticdnn.model.cnn import ElasticCNNUtil
    # raw_cnn = resnet50()
    # prunable_layers = []
    # for i in range(1, 5):
    #     for j in range([3, 4, 6, 3][i - 1]):
    #         prunable_layers += [f'layer{i}.{j}.conv1', f'layer{i}.{j}.conv2']
    # ignore_layers = [layer for layer, m in raw_cnn.named_modules() if isinstance(m, nn.Conv2d) and layer not in prunable_layers]
    # test(raw_cnn, ignore_layers, ElasticCNNUtil(), torch.rand(1, 3, 224, 224))
    ignore_layers = []
    from methods.elasticdnn.model.vit import ElasticViTUtil
    # raw_vit = torch.load('tmp-master-dnn.pt')
    raw_vit = torch.load('')
    test(raw_vit, ignore_layers, ElasticViTUtil(), torch.rand(16, 3, 224, 224).cuda(), 0.9)
    exit()
    
    
    from dnns.vit import vit_b_16
    # from methods.elasticdnn.model.vit_new import ElasticViTUtil
    from methods.elasticdnn.model.vit import ElasticViTUtil
    # raw_vit = vit_b_16()
    
    for s in [0.8, 0.9, 0.95]:
        raw_vit = vit_b_16().cuda()
        
        ignore_layers = []
        test(raw_vit, ignore_layers, ElasticViTUtil(), torch.rand(16, 3, 224, 224).cuda(), s)
    
    # for s in [0, 0.2, 0.4, 0.6, 0.8]:
    #     pretrained_md_models_dict_path = 'experiments/elasticdnn/vit_b_16/offline/fm_to_md/results/20230518/999999-164524-wo_FBS_trial_dsnet_lr/models/md_best.pt'
    #     raw_vit = torch.load(pretrained_md_models_dict_path)['main'].cuda()
        
    #     ignore_layers = []
    #     test(raw_vit, ignore_layers, ElasticViTUtil(), torch.rand(16, 3, 224, 224).cuda(), s)
    # exit()
    
    
    # weight = torch.rand((10, 5))
    # bias = torch.rand(10)
    # x = torch.rand((1, 3, 5))
    
    # t = torch.randperm(5)
    # pruned, unpruned = t[0: 3], t[3: ]

    # mask = torch.ones_like(x)
    # mask[:, :, pruned] = 0
    
    # print(x, x * mask, (x * mask).sum((0, 1)))

    # import torch.nn.functional as F
    # o1 = F.linear(x * mask, weight, bias)
    # # print(o1)
    
    
    # o2 = F.linear(x[:, :, unpruned], weight[:, unpruned], bias)
    # # print(o2)
    
    # print(o1.size(), o2.size(), ((o1 - o2) ** 2).sum())
    
    
    
    
    # weight = torch.rand((130, 5))
    # bias = torch.rand(130)
    # x = torch.rand((1, 3, 5))
    
    # t = torch.randperm(5)
    # pruned, unpruned = t[0: 3], t[3: ]

    # mask = torch.ones_like(x)
    # mask[:, :, pruned] = 0
    
    # print(x, x * mask, (x * mask).sum((0, 1)))

    # import torch.nn.functional as F
    # o1 = F.linear(x * mask, weight, bias)
    # # print(o1)
    
    
    # o2 = F.linear(x[:, :, unpruned], weight[:, unpruned], bias)
    # # print(o2)
    
    # print(o1.size(), o2.size(), ((o1 - o2) ** 2).sum())
    
    
    
    
    
    # weight = torch.rand((1768, 768))
    # bias = torch.rand(1768)
    # x = torch.rand([1, 197, 768])
    
    # t = torch.randperm(768)
    # unpruned, pruned = t[0: 144], t[144: ]
    # unpruned = unpruned.sort()[0]
    # pruned = pruned.sort()[0]

    # mask = torch.ones_like(x)
    # mask[:, :, pruned] = 0
    
    # print(x.sum((0, 1)).size(), (x * mask).sum((0, 1))[0: 10], x[:, :, unpruned].sum((0, 1))[0: 10])

    # import torch.nn.functional as F
    # o1 = F.linear(x * mask, weight, bias)
    # o2 = F.linear(x[:, :, unpruned], weight[:, unpruned], bias)
    # print(o1.sum((0, 1))[0: 10], o2.sum((0, 1))[0: 10], o1.size(), o2.size(), ((o1 - o2).abs()).sum(), ((o1 - o2) ** 2).sum())
    # unpruned_indexes = torch.randperm(5)[0: 2]
    # o2 = F.linear(x[:, unpruned_indexes], weight[:, unpruned_indexes])
    # print(o2)