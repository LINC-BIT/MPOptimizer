from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
import torch


def test(fm, fm_to_md_util: FM_to_MD_Util, samples: torch.Tensor):
    master_dnn = fm_to_md_util.init_md_from_fm_by_reducing_width_with_perf_test(fm, 4, samples)
    torch.save(master_dnn, 'tmp-master-dnn.pt')
    
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
    # test(raw_cnn, ignore_layers, ElasticCNNUtil(), torch.rand(2, 3, 224, 224))
    
    
    from dnns.vit import vit_b_16
    from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
    raw_vit = vit_b_16()
    test(raw_vit.cuda(), FM_to_MD_ViT_Util(), torch.rand(2, 3, 224, 224).cuda())
    