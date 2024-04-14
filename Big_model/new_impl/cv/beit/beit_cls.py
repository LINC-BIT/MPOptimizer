import torch
from new_impl.cv.elasticdnn.api.model import ElasticDNN_OfflineClsFMModel
#from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from new_impl.cv.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from new_impl.cv.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from beit import FMLoRA_beit_Util
from new_impl.cv.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util

from new_impl.cv.elasticdnn.model.vit import ElasticViTUtil
from data import build_scenario
import torch.nn.functional as F
from utils.dl.common.model import LayerActivation, get_module
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
# from transformers import CvtForImageClassification
# model = CvtForImageClassification.from_pretrained("/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/cvt_model",num_labels=20,ignore_mismatched_sizes=True).to('cuda')

class ElasticDNN_beit_OfflineClsFMModel(ElasticDNN_OfflineClsFMModel):
    def generate_md_by_reducing_width(self, reducing_width_ratio, samples: torch.Tensor):
        # return FM_to_MD_ViT_Util().init_md_from_fm_by_reducing_width_with_perf_test(self.models_dict['main'], 
        #                                                                 reducing_width_ratio, samples)
        raise NotImplementedError
    def get_feature_hook(self) -> LayerActivation:
        return LayerActivation(get_module(self.models_dict['main'], 'classifier'), True, self.device)
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        raise NotImplementedError
    
    def forward_to_get_task_loss(self, x, y, *args, **kwargs):
        #x1 = torch.rand(1,3,224,224).to('cuda:1')
        o1 = self.infer(x)
        # o2 = self.infer(x1)
        # print(o1.logits)
        # print(o2.logits)
        #print(self.models_dict['main'])
        #print(o1.logits.shape)
        #print(F.cross_entropy(self.infer(x).logits, y) )
        #formatted_values = [[round(value, 4) for value in row] for row in o1.logits.tolist()]
        #return F.cross_entropy(torch.tensor(formatted_values).to('cuda'), y)
        return F.cross_entropy(o1.logits, y) #这个是适用于hugging face模型的计算形式，因为它输出的是一个实例化的类，结果封装在类的属性里，你得去给它调出来。
    
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_beit_Util()
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['main'], 'classifier')
        return list(head.parameters())
if __name__ == '__main__':
    scenario = build_scenario(
        source_datasets_name=['GTA5Cls', 'SuperviselyPersonCls'],
        target_datasets_order=['CityscapesCls', 'BaiduPersonCls'] * 15,
        da_mode='close_set',
        data_dirs={
            'GTA5Cls': '/data/zql/datasets/gta5_for_cls_task',
            'SuperviselyPersonCls': '/data/zql/datasets/supervisely_person_for_cls_task',
            'CityscapesCls': '/data/zql/datasets/cityscapes_for_cls_task',
            'BaiduPersonCls': '/data/zql/datasets/baidu_person_for_cls_task'
        },
    )

    from transformers import BeitForImageClassification
    fm_models_dict_path = save_models_dict_for_init({
        'main':BeitForImageClassification.from_pretrained('new_impl/cv/beit/beit_model',num_labels=scenario.num_classes,ignore_mismatched_sizes=True)
    },__file__,'dinat_pretrained')
    torch.cuda.set_device(1)
    device = 'cuda'
    #print(CvtForImageClassification.from_pretrained("/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/cvt_model",num_labels=scenario.num_classes,ignore_mismatched_sizes=True))
    fm_model = ElasticDNN_beit_OfflineClsFMModel('fm', fm_models_dict_path, device)
    #fm_model = CvtForImageClassification.from_pretrained("/data/zql/concept-drift-in-edge-projects/UniversalElasticNet/new_impl/cv/cvt_model",num_labels=scenario.num_classes,ignore_mismatched_sizes=True).to(device)
    models = {
        'fm':fm_model
    }
    import sys
    fm_lora_alg = ElasticDNN_FMLoRAAlg(models, get_res_save_dir(__file__, tag=sys.argv[0]))
    
    from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
    fm_lora_alg.run(scenario, hyps={
        'launch_tbboard': False,
        
        'samples_size': (1, 3, 224, 224),
        'ab_r': 3,#hugging face中的模型封装得特别严实，自注意力层里面，qkv是分开的，注意这个对应的层数不要设置太高
        'train_batch_size': 256,
        'val_batch_size': 512,
        'num_workers': 16,
        'optimizer': 'Adam',
        'optimizer_args': {'lr': 1e-3, 'betas': [0.9, 0.999]},#不同的模型，注意调调学习率啊
        'scheduler': 'LambdaLR',
        'scheduler_args': {'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)},
        'num_iters': 8000,
        'val_freq': 400
    }
)
