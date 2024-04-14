import torch
import torch.nn as nn
from methods.elasticdnn.api.algs.fm_lora import ElasticDNN_FMLoRAAlg
from methods.elasticdnn.api.algs.md_pretraining_wo_fbs import ElasticDNN_MDPretrainingWoFBSAlg
from methods.elasticdnn.model.base import ElasticDNNUtil
from methods.elasticdnn.pipeline.offline.fm_lora.base import FMLoRA_Util
from glip import ElasticGLIPUtil, FMLoRA_GLIP_Util, FM_to_MD_GLIP_Util, ElasticDNN_OfflineMMDetFMModel, ElasticDNN_OfflineMMDetMDModel
from methods.elasticdnn.pipeline.offline.fm_to_md.base import FM_to_MD_Util
from methods.elasticdnn.pipeline.offline.fm_to_md.vit import FM_to_MD_ViT_Util
from methods.elasticdnn.model.vit import ElasticViTUtil
from methods.elasticdnn.api.algs.md_pretraining_index_v2_train_index_and_md import ElasticDNN_MDPretrainingIndexAlg
from utils.dl.common.model import LayerActivation3, get_module, get_parameter
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data import build_scenario
import torch.nn.functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList
import os
from utils.dl.common.loss import CrossEntropyLossSoft
from new_impl.cv.feat_align.main_glip import OnlineFeatAlignModel, FeatAlignAlg
import tqdm
from new_impl.cv.feat_align.mmd import mmd_rbf
from new_impl.cv.utils.baseline_da import baseline_da
from new_impl.cv.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

torch.cuda.set_device(0)
device = 'cuda'
app_name = 'cls'

scenario = build_scenario(
        source_datasets_name=['MM-COCO2017'],
        target_datasets_order=['MM-CityscapesDet', 'MM-GTA5Det'] * 10,
        da_mode='close_set',
        data_dirs={
            'MM-COCO2017': '/data/zql/datasets/coco2017',
            'MM-CityscapesDet': '/data/zql/datasets/cityscape',
            'MM-GTA5Det': '/data/zql/datasets/GTA-ls-copy/GTA5',
        },
    )
    
class DetOnlineFeatAlignModel(OnlineFeatAlignModel):
    def get_trained_params(self): # TODO: elastic fm only train a part of params
        #qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters() if 'attention.attention.projection_query' in n or 'attention.attention.projection_key' in n or 'attention.attention.projection_value' in n or 'intermediate.dense' in n or 'output.dense' in n]
        qkv_and_norm_params = [p for n, p in self.models_dict['main'].named_parameters()]
        return qkv_and_norm_params
    
    def get_feature_hook(self):
        return LayerActivation3(get_module(self.models_dict['main'], 'model.rpn'), False, self.device)
    
    def forward_to_get_task_loss(self, x, y):
        loss_dict = self.infer(x)
        losses = sum(loss for loss in loss_dict.values())
        # print(losses)
        
        return losses
    
    def get_mmd_loss(self, f1, f2):
        # f1 = f1.view(f1.shape[0], -1)
        # f2 = f2.view(f2.shape[0], -1)
        return mmd_rbf(f1, f2)
    
    def infer(self, x, *args, **kwargs):
        return self.models_dict['main'](**x)
    
    def get_accuracy(self, test_loader, *args, **kwargs):
        # print('DeeplabV3: start test acc')
        _d = test_loader.dataset
        imgsz = _d.cocods.img_size
        cls_num = len(_d.cocods.class_ids)
        # num_classes = len(_d.cls_names)
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
                    img_size=imgsz,
                    confthre=0.01,
                    nmsthre=0.65,
                    num_classes=cls_num,
                    testdev=False
                )
                res = evaluator.evaluate(model, False, False, decoder=MMCOCODecoder)
                map50 = res[1]
            # print('eval info', res[-1])
        return map50

from glip import glip_model, build_transform, run_ner, collect_mm_fn
cfg_path = 'new_impl/cv/glip/object_detection/pretrained_model/glip_Swin_T_O365_GoldG.yaml'
model_path = 'new_impl/cv/glip/object_detection/pretrained_model/glip_tiny_model_o365_goldg_cc_sbu.pth'
config, _ = glip_model(cfg_path, model_path)
transform = build_transform(config, None)

da_alg = FeatAlignAlg
from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
#from new_impl.cv.model import ClsOnlineFeatAlignModel
da_model = DetOnlineFeatAlignModel(
    app_name,
    'new_impl/cv/glip/object_detection/results/det_md_wo_fbs.py/20231129/999999-153230-results/models/md_best.pt',
    device
)
da_alg_hyp = {
    'MM-GTA5Det': {
        'train_batch_size': 8,
        'val_batch_size': 1,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 2e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'feat_align_loss_weight': 0.3,
        'transform':transform
    },
    'MM-CityscapesDet': {
        'train_batch_size': 8,
        'val_batch_size': 1,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 2e-6, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'feat_align_loss_weight': 0.3,
        'transform':transform
    },
}


baseline_da(
    app_name,
    scenario,
    da_alg,
    da_alg_hyp,
    da_model,
    device,
    __file__,
    "results",
    collate_fn=collect_mm_fn
)