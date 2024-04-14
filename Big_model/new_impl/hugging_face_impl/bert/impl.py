from methods.elasticdnn.hugging_face.user_impl import HuggingFaceModelAPI
from utils.dl.common.model import LayerActivation, get_module
import torch
import torch.nn.functional as F
from torch import nn
import tqdm


class BERTHuggingFaceModelAPI(HuggingFaceModelAPI):
    def get_feature_hook(self, fm: nn.Module, device) -> LayerActivation:
        return LayerActivation(get_module(fm, 'classifier'), True, device)
    
    def get_task_head_params(self, fm: nn.Module):
        head = get_module(fm, 'classifier')
        return list(head.parameters())
    
    def get_qkv_proj_ff1_ff2_layer_names(self):
        return [[f'bert.encoder.layer.{i}.attention.self.query', f'bert.encoder.layer.{i}.attention.self.key', f'bert.encoder.layer.{i}.attention.self.value', \
            f'bert.encoder.layer.{i}.attention.output.dense', \
            f'bert.encoder.layer.{i}.intermediate.dense', f'bert.encoder.layer.{i}.output.dense'] for i in range(12)]
    
    def get_accuracy(self, fm: nn.Module, test_loader, device, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        fm.eval()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
                y = y.to(device)
                output = self.infer(fm, x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                
                # print(pred, y)
                
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
    
    def infer(self, fm: nn.Module, x, *args, **kwargs):
        return fm(**x)
    
    def forward_to_get_task_loss(self, fm: nn.Module, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(fm, x), y)