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
from new_impl.cv.utils.elasticfm_da import init_online_model, elasticfm_da
from new_impl.cv.elasticdnn.api.online_model_v2 import ElasticDNN_OnlineModel
from utils.common.log import logger

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

torch.cuda.set_device(0)
device = 'cuda'
app_name = 'cls'
sd_sparsity = 0.8

settings = {
    'involve_fm': True
}

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

class ElasticDNN_DetOnlineModel(ElasticDNN_OnlineModel):
    def get_accuracy(self, test_loader, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        self.to_eval_mode()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)
                output = self.infer(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
    
    def get_elastic_dnn_util(self) -> ElasticDNNUtil:
        return ElasticGLIPUtil()
    
    def get_fm_matched_param_of_md_param(self, md_param_name):
        # only between qkv.weight, norm.weight/bias
        self_param_name = md_param_name
        fm = self.models_dict['fm']
        # if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
        #     return None
        
        # p = get_parameter(self.models_dict['md'], self_param_name)
        # if p.dim() == 0:
        #     return None
        # elif p.dim() == 1 and 'norm' in self_param_name and 'weight' in self_param_name:
        #     return get_parameter(fm, self_param_name)
        
        if any([k in self_param_name for k in ['fbs', 'cls_token', 'pos_embed']]):
            return None
        
        p = get_parameter(self.models_dict['md'], self_param_name)
        if p.dim() == 0:
            return None
        # elif p.dim() == 1 and 'layernorm' in self_param_name and 'weight' in self_param_name:
        #     return get_parameter(fm, self_param_name)
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        # if 'qkv.weight' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -1]) + '.qkv'
        #     fm_qkv = get_module(fm, fm_qkv_name)
            
        #     fm_abs_name = '.'.join(ss[0: -1]) + '.abs'
        #     fm_abs = get_module(fm, fm_abs_name)
            
        #     # NOTE: unrecoverable operation! multiply LoRA parameters to allow it being updated in update_fm_param()
        #     # TODO: if fm will be used for inference, _mul_lora_weight will not be applied!
        #     if not hasattr(fm_abs, '_mul_lora_weight'):
        #         logger.debug(f'set _mul_lora_weight in {fm_abs_name}')
        #         setattr(fm_abs, '_mul_lora_weight', 
        #                 nn.Parameter(torch.cat([(_abs[0].weight.T @ _abs[1].weight.T).T for _abs in fm_abs], dim=0)))
            
        #     return torch.cat([
        #         fm_qkv.weight.data, # task-agnositc params
        #         fm_abs._mul_lora_weight.data # task-specific params (LoRA)
        #     ], dim=0)
            
        # # elif 'to_qkv.bias' in self_param_name:
        # #     ss = self_param_name.split('.')
            
        # #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        # #     return get_parameter(fm, fm_qkv_name)
            
        # elif 'mlp.fc1' in self_param_name and 'weight' in self_param_name:
        #     fm_param_name = self_param_name.replace('.linear', '')
        #     return get_parameter(fm, fm_param_name)

        # elif 'mlp.fc2' in self_param_name and 'weight' in self_param_name:
        #     fm_param_name = self_param_name
        #     return get_parameter(fm, fm_param_name)
        
        # else:
        #     # return get_parameter(fm, self_param_name)
        #     return None
        if ('attn.qkv' in self_param_name or \
        'attn.v_proj' in self_param_name or 'attn.l_proj' in self_param_name or 'attn.values_v_proj' in self_param_name or 'attn.values_l_proj' in self_param_name) and ('weight' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            if not hasattr(fm_abs, '_mul_lora_weight'):
                logger.debug(f'set _mul_lora_weight in {fm_abs_name}')
                setattr(fm_abs, '_mul_lora_weight', 
                        nn.Parameter(fm_abs[1].weight @ fm_abs[0].weight))
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs._mul_lora_weight.data # task-specific params (LoRA)
            ], dim=0)
            
        elif ('attn.qkv' in self_param_name or \
        'attn.v_proj' in self_param_name or 'attn.l_proj' in self_param_name or 'attn.values_v_proj' in self_param_name or 'attn.values_l_proj' in self_param_name) and ('bias' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc.bias'
            return get_parameter(fm, fm_qkv_name)
        
        elif ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('weight' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            
            if not hasattr(fm_abs, '_mul_lora_weight'):
                logger.debug(f'set _mul_lora_weight in {fm_abs_name}')
                setattr(fm_abs, '_mul_lora_weight', 
                        nn.Parameter(fm_abs[1].weight @ fm_abs[0].weight))
            
            return torch.cat([
                fm_qkv.weight.data, # task-agnositc params
                fm_abs._mul_lora_weight.data # task-specific params (LoRA)
            ], dim=0)
            
        elif ('query' in self_param_name or 'key' in self_param_name or \
            'value' in self_param_name) and ('bias' in self_param_name):
            ss = self_param_name.split('.')
            
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc.bias'
            return get_parameter(fm, fm_qkv_name)
        
        elif 'mlp.fc1' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name.replace('.linear', '')
            return get_parameter(fm, fm_param_name)
        # elif 'mlp.fc2' in self_param_name and 'weight' in self_param_name:
        #     fm_param_name = self_param_name.replace('.linear', '')
        #     return get_parameter(fm, fm_param_name)
        else:
            #return get_parameter(fm, self_param_name)
            return None


    def update_fm_param(self, md_param_name, cal_new_fm_param_by_md_param):
        if not ('attn.qkv.weight' in md_param_name or 'attn.v_proj.weight' in md_param_name or \
                'attn.l_proj.weight' in md_param_name or 'attn.values_v_proj.weight' in md_param_name or \
                'attn.values_l_proj.weight' in md_param_name or 'query.weight' in md_param_name or 'key.weight' in md_param_name or \
            'value.weight' in md_param_name):
            matched_fm_param_ref = self.get_fm_matched_param_of_md_param(md_param_name)
            matched_fm_param_ref.copy_(cal_new_fm_param_by_md_param)
        else:
            new_fm_attn_weight, new_fm_lora_weight = torch.chunk(cal_new_fm_param_by_md_param, 2, 0)
            ss = md_param_name.split('.')
            fm = self.models_dict['fm']
            # update task-agnostic parameters
            fm_qkv_name = '.'.join(ss[0: -1]) + '.fc'
            fm_qkv = get_module(fm, fm_qkv_name)
            fm_qkv.weight.data.copy_(new_fm_attn_weight)
            
            # update task-specific parameters
            fm_abs_name = '.'.join(ss[0: -1]) + '.ab'
            fm_abs = get_module(fm, fm_abs_name)
            fm_abs._mul_lora_weight.data.copy_(new_fm_lora_weight) # TODO: this will not be applied in inference!
        
    def get_md_matched_param_of_fm_param(self, fm_param_name):
        return super().get_md_matched_param_of_fm_param(fm_param_name)
    
    def get_md_matched_param_of_sd_param(self, sd_param_name):
        # raise NotImplementedError

        # only between qkv.weight, norm.weight/bias
        self_param_name = sd_param_name
        md = self.models_dict['md']
        if any([k in self_param_name for k in ['fbs', 'ab', 'embeddings']]):
            return None
        
        p = get_parameter(self.models_dict['sd'], self_param_name)
        if p.dim() == 0:
            return None
        elif p.dim() == 1 and ('LayerNorm' in self_param_name or 'layernorm' in self_param_name) and 'weight' in self_param_name:
            return get_parameter(md, self_param_name)
        
        # 1. xx.qkv.to_qkv.yy to xx.qkv.qkv.aa and xx.qkv.abs.zz
        if ('attn.qkv' in sd_param_name or 'attn.v_proj' in sd_param_name or \
            'attn.l_proj' in sd_param_name or 'attn.values_v_proj' in sd_param_name or \
            'attn.values_l_proj' in sd_param_name or 'query' in sd_param_name or 'key' in sd_param_name or \
            'value' in sd_param_name) and ('weight' in self_param_name):
            
        
            return get_parameter(md, self_param_name) # NOTE: no fbs in qkv!
            
        # elif 'to_qkv.bias' in self_param_name:
        #     ss = self_param_name.split('.')
            
        #     fm_qkv_name = '.'.join(ss[0: -2]) + '.qkv.bias'
        #     return get_parameter(fm, fm_qkv_name)
            
        elif 'mlp.fc1.0.weight' in self_param_name:
            fm_param_name = '.'.join(self_param_name.split('.')[0: -2]) + '.linear.weight'
            return get_parameter(md, fm_param_name)

        elif 'mlp.fc2' in self_param_name and 'weight' in self_param_name:
            fm_param_name = self_param_name
            return get_parameter(md, fm_param_name)
        
        else:
            # return get_parameter(fm, self_param_name)
            return None
    
    def get_task_head_params(self):
        head = get_module(self.models_dict['sd'], 'classifier')
        return list(head.parameters())
    
    
    
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




#from new_impl.cv.model import ElasticDNN_ClsOnlineModel
elasticfm_model = ElasticDNN_DetOnlineModel('det', init_online_model(
    'new_impl/cv/glip/object_detection/results/det_md_w_fbs_index.py/20231201/999996-195158-results/models/fm_best.pt',
    'new_impl/cv/glip/object_detection/results/det_md_w_fbs_index.py/20231201/999996-195158-results/models/md_best.pt',
    'det', __file__
), device, {
    'md_to_fm_alpha': 0.1,
    'fm_to_md_alpha': 0.1
})

from glip import glip_model, build_transform, run_ner, collect_mm_fn
cfg_path = 'new_impl/cv/glip/object_detection/pretrained_model/glip_Swin_T_O365_GoldG.yaml'
model_path = 'new_impl/cv/glip/object_detection/pretrained_model/glip_tiny_model_o365_goldg_cc_sbu.pth'
config, _ = glip_model(cfg_path, model_path)
transform = build_transform(config, None)

da_alg = FeatAlignAlg
from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
#from new_impl.cv.model import ClsOnlineFeatAlignModel
da_model = DetOnlineFeatAlignModel
da_alg_hyp = {
    'MM-GTA5Det': {
        'train_batch_size': 8,
        'val_batch_size': 1,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 5e-7, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'sd_sparsity':0.3,
        'feat_align_loss_weight': 0.0,
        'transform':transform
    },
    'MM-CityscapesDet': {
        'train_batch_size': 8,
        'val_batch_size': 1,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'optimizer_args': {'lr': 5e-7, 'betas': [0.9, 0.999], 'weight_decay': 0.01},
        'scheduler': '',
        'scheduler_args': {},
        'num_iters': 100,
        'val_freq': 20,
        'sd_sparsity':0.3,
        'feat_align_loss_weight': 0.0,
        'transform':transform
    },
}


elasticfm_da(
    [app_name],
    [scenario],
    [elasticfm_model],
    [da_alg],
    [da_alg_hyp],
    [da_model],
    device,
    settings,
    __file__,
    "results",
    collate_fn=collect_mm_fn
)