from methods.elasticdnn.hugging_face.user_impl import HuggingFaceModelAPI
from utils.dl.common.model import LayerActivation, get_module
import torch
import torch.nn.functional as F
from torch import nn
import tqdm


class ViTHuggingFaceModelAPI(HuggingFaceModelAPI):
    def get_feature_hook(self, fm: nn.Module, device) -> LayerActivation:
        return LayerActivation(get_module(fm, 'head'), True, device)
    
    def get_task_head_params(self, fm: nn.Module):
        head = get_module(fm, 'head')
        return list(head.parameters())
    
    def get_qkv_proj_ff1_ff2_layer_names(self):
        return [[f'blocks.{i}.attn.qkv', f'blocks.{i}.attn.proj', f'blocks.{i}.mlp.fc1', f'blocks.{i}.mlp.fc2', ] for i in range(12)]
    
    def get_accuracy(self, fm: nn.Module, test_loader, device, *args, **kwargs):
        acc = 0
        sample_num = 0
        
        fm.eval()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True, leave=False)
            for batch_index, (x, y) in pbar:
                x, y = x.to(device), y.to(device)
                output = fm(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)
                
                pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
                                     f'cur_batch_acc: {(correct / len(y)):.4f}')

        acc /= sample_num
        return acc
    
    def infer(self, fm: nn.Module, x, *args, **kwargs):
        return fm(x)
    
    def forward_to_get_task_loss(self, fm: nn.Module, x, y, *args, **kwargs):
        return F.cross_entropy(self.infer(fm, x), y)