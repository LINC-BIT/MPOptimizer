from abc import abstractmethod
from copy import deepcopy
import enum
import torch
from torch import nn
import os

from .model_fbs import DomainDynamicConv2d
#from methods.utils.data import get_source_dataloader, get_source_normal_aug_dataloader, get_target_dataloaders
#from models.resnet_cifar.model_manager import ResNetCIFARManager
from utils.common.others import get_cur_time_str
from utils.dl.common.env import set_random_seed
from utils.dl.common.model import get_model_latency, get_model_size, get_module, set_module
from utils.common.log import logger
from utils.third_party.nni_new.compression.pytorch.speedup import ModelSpeedup
from utils.third_party.nni_new.compression.pytorch.utils.mask_conflict import GroupMaskConflict, ChannelMaskConflict, CatMaskPadding


def fix_mask_conflict(masks, model=None, dummy_input=None, traced=None, fix_group=False, fix_channel=True, fix_padding=False):
    if isinstance(masks, str):
        # if the input is the path of the mask_file
        assert os.path.exists(masks)
        masks = torch.load(masks)
    assert len(masks) > 0, 'Mask tensor cannot be empty'
    # if the user uses the model and dummy_input to trace the model, we
    # should get the traced model handly, so that, we only trace the
    # model once, GroupMaskConflict and ChannelMaskConflict will reuse
    # this traced model.
    if traced is None:
        assert model is not None and dummy_input is not None
        training = model.training
        model.eval()
        # We need to trace the model in eval mode
        traced = torch.jit.trace(model, dummy_input)
        model.train(training)

    if fix_group:
        fix_group_mask = GroupMaskConflict(masks, model, dummy_input, traced)
        masks = fix_group_mask.fix_mask()
    if fix_channel:
        fix_channel_mask = ChannelMaskConflict(masks, model, dummy_input, traced)
        masks = fix_channel_mask.fix_mask()
    if fix_padding:
        padding_cat_mask = CatMaskPadding(masks, model, dummy_input, traced)
        masks = padding_cat_mask.fix_mask()
    return masks


class FeatureBoosting(nn.Module):
    def __init__(self, w: torch.Tensor):
        super(FeatureBoosting, self).__init__()
        assert w.dim() == 1
        self.w = nn.Parameter(w.unsqueeze(0).unsqueeze(2).unsqueeze(3), requires_grad=False)
        
    def forward(self, x):
        return x * self.w


class FBSSubModelExtractor:
    def extract_submodel_via_a_sample(self, fbs_model: nn.Module, sample: torch.Tensor):
        assert sample.dim() == 4 and sample.size(0) == 1
        
        fbs_model.eval()
        o1 = fbs_model(sample)
        
        pruning_info = {}
        pruning_masks = {}
        
        for layer_name, layer in fbs_model.named_modules():
            if not isinstance(layer, DomainDynamicConv2d):
                continue
            
            cur_pruning_mask = {'weight': torch.zeros_like(layer.raw_conv2d.weight.data)}
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'] = torch.zeros_like(layer.raw_conv2d.bias.data)
            
            w = get_module(fbs_model, layer_name).cached_w.squeeze()
            unpruned_filters_index = w.nonzero(as_tuple=True)[0]
            pruning_info[layer_name] = w
            
            cur_pruning_mask['weight'][unpruned_filters_index, ...] = 1.
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'][unpruned_filters_index, ...] = 1.
            pruning_masks[layer_name + '.0'] = cur_pruning_mask
        
        no_gate_model = deepcopy(fbs_model)
        for name, layer in no_gate_model.named_modules():
            if not isinstance(layer, DomainDynamicConv2d):
                continue
            # layer.bn.weight.data.mul_(pruning_info[name])
            set_module(no_gate_model, name, nn.Sequential(layer.raw_conv2d, layer.bn, nn.Identity()))
            
        # fixed_pruning_masks = fix_mask_conflict(pruning_masks, fbs_model, sample.size(), None, True, True, True)
        tmp_mask_path = f'tmp_mask_{get_cur_time_str()}_{os.getpid()}.pth'
        torch.save(pruning_masks, tmp_mask_path)
        pruned_model = no_gate_model
        pruned_model.eval()
        model_speedup = ModelSpeedup(pruned_model, sample, tmp_mask_path, sample.device)
        model_speedup.speedup_model()
        os.remove(tmp_mask_path)
        
        # add feature boosting module
        for layer_name, feature_boosting_w in pruning_info.items():
            feature_boosting_w = feature_boosting_w[feature_boosting_w.nonzero(as_tuple=True)[0]]
            set_module(pruned_model, layer_name + '.2', FeatureBoosting(feature_boosting_w))
        
        pruned_model_size = get_model_size(pruned_model, True)
        pruned_model.eval()
        o2 = pruned_model(sample)
        diff = ((o1 - o2) ** 2).sum()
        logger.info(f'pruned model size: {pruned_model_size:.3f}MB, diff: {diff}')
        
        return pruned_model
    
    @abstractmethod
    def get_final_w(self, fbs_model: nn.Module, samples: torch.Tensor, layer_name: str, w: torch.Tensor):
        pass
    
    @abstractmethod
    def generate_pruning_strategy(self, fbs_model: nn.Module, samples: torch.Tensor):
        pass
    
    def extract_submodel_via_samples(self, fbs_model: nn.Module, samples: torch.Tensor):
        assert samples.dim() == 4
        fbs_model = deepcopy(fbs_model)
        # fbs_model.eval()
        # fbs_model(samples)
        self.generate_pruning_strategy(fbs_model, samples)
        
        pruning_info = {}
        pruning_masks = {}
        
        for layer_name, layer in fbs_model.named_modules():

            if not isinstance(layer, DomainDynamicConv2d):
                continue
            
            cur_pruning_mask = {'weight': torch.zeros_like(layer.raw_conv2d.weight.data)}
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'] = torch.zeros_like(layer.raw_conv2d.bias.data)
            
            w = get_module(fbs_model, layer_name).cached_w.squeeze() # 2-dim
            w = self.get_final_w(fbs_model, samples, layer_name, w)
            
            unpruned_filters_index = w.nonzero(as_tuple=True)[0]
            pruning_info[layer_name] = w
            
            cur_pruning_mask['weight'][unpruned_filters_index, ...] = 1.
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'][unpruned_filters_index, ...] = 1.
            pruning_masks[layer_name + '.0'] = cur_pruning_mask
        
        no_gate_model = deepcopy(fbs_model)
        for name, layer in no_gate_model.named_modules():
            if not isinstance(layer, DomainDynamicConv2d):
                continue
            # layer.bn.weight.data.mul_(pruning_info[name])
            set_module(no_gate_model, name, nn.Sequential(layer.raw_conv2d, layer.bn, nn.Identity()))
            
        # fixed_pruning_masks = fix_mask_conflict(pruning_masks, fbs_model, sample.size(), None, True, True, True)
        tmp_mask_path = f'tmp_mask_{get_cur_time_str()}_{os.getpid()}.pth'
        torch.save(pruning_masks, tmp_mask_path)
        pruned_model = no_gate_model
        pruned_model.eval()
        model_speedup = ModelSpeedup(pruned_model, samples[0:1], tmp_mask_path, samples.device)
        model_speedup.speedup_model()
        os.remove(tmp_mask_path)
        
        # add feature boosting module
        for layer_name, feature_boosting_w in pruning_info.items():
            feature_boosting_w = feature_boosting_w[feature_boosting_w.nonzero(as_tuple=True)[0]]
            set_module(pruned_model, layer_name + '.2', FeatureBoosting(feature_boosting_w))
        
        return pruned_model, pruning_info
    
    def extract_submodel_via_samples_and_last_submodel(self, fbs_model: nn.Module, samples: torch.Tensor, 
                                                       last_submodel: nn.Module, last_pruning_info: dict):
        assert samples.dim() == 4
        fbs_model = deepcopy(fbs_model)
        # fbs_model.eval()
        # fbs_model(samples)
        self.generate_pruning_strategy(fbs_model, samples)
        
        pruning_info = {}
        pruning_masks = {}
        # some tricks
        incrementally_updated_layers = []
        
        for layer_name, layer in fbs_model.named_modules():
            if not isinstance(layer, DomainDynamicConv2d):
                continue
            
            cur_pruning_mask = {'weight': torch.zeros_like(layer.raw_conv2d.weight.data)}
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'] = torch.zeros_like(layer.raw_conv2d.bias.data)
            
            w = get_module(fbs_model, layer_name).cached_w.squeeze() # 2-dim
            w = self.get_final_w(fbs_model, samples, layer_name, w)
            
            unpruned_filters_index = w.nonzero(as_tuple=True)[0]
            pruning_info[layer_name] = w
            
            cur_pruning_mask['weight'][unpruned_filters_index, ...] = 1.
            if layer.raw_conv2d.bias is not None:
                cur_pruning_mask['bias'][unpruned_filters_index, ...] = 1.
            pruning_masks[layer_name + '.0'] = cur_pruning_mask
            
            # some tricks
            if last_pruning_info is not None:
                last_w = last_pruning_info[layer_name]
                intersection_ratio = ((w > 0) * (last_w > 0)).sum() / (last_w > 0).sum()
                if intersection_ratio > 0.:
                    incrementally_updated_layers += [layer_name] # that is, only similar layers are transferable
        
        no_gate_model = deepcopy(fbs_model)
        for name, layer in no_gate_model.named_modules():
            if not isinstance(layer, DomainDynamicConv2d):
                continue
            # layer.bn.weight.data.mul_(pruning_info[name])
            set_module(no_gate_model, name, nn.Sequential(layer.raw_conv2d, layer.bn, nn.Identity()))
            
        # fixed_pruning_masks = fix_mask_conflict(pruning_masks, fbs_model, sample.size(), None, True, True, True)
        tmp_mask_path = f'tmp_mask_{get_cur_time_str()}_{os.getpid()}.pth'

        torch.save(pruning_masks, tmp_mask_path)
        pruned_model = no_gate_model
        pruned_model.eval()
        model_speedup = ModelSpeedup(pruned_model, samples[0:1], tmp_mask_path, samples.device)
        model_speedup.speedup_model()
        os.remove(tmp_mask_path)
        
        # add feature boosting module
        for layer_name, feature_boosting_w in pruning_info.items():
            feature_boosting_w = feature_boosting_w[feature_boosting_w.nonzero(as_tuple=True)[0]]
            set_module(pruned_model, layer_name + '.2', FeatureBoosting(feature_boosting_w))
            
        # some tricks
        # incrementally updating (borrow some weights from last_pruned_model)
        for layer_name in incrementally_updated_layers:
            cur_filter_i, last_filter_i = 0, 0
            for i, (w_factor, last_w_factor) in enumerate(zip(pruning_info[layer_name], last_pruning_info[layer_name])):
                if w_factor > 0 and last_w_factor > 0: # the filter is shared
                    cur_conv2d, last_conv2d = get_module(pruned_model, layer_name + '.0'), get_module(last_submodel, layer_name + '.0')
                    cur_conv2d.weight.data[cur_filter_i] = last_conv2d.weight.data[last_filter_i]
                    
                    cur_bn, last_bn = get_module(pruned_model, layer_name + '.1'), get_module(last_submodel, layer_name + '.1')
                    cur_bn.weight.data[cur_filter_i] = last_bn.weight.data[last_filter_i]
                    cur_bn.bias.data[cur_filter_i] = last_bn.bias.data[last_filter_i]
                    cur_bn.running_mean.data[cur_filter_i] = last_bn.running_mean.data[last_filter_i]
                    cur_bn.running_var.data[cur_filter_i] = last_bn.running_var.data[last_filter_i]
                    
                    cur_fw, last_fw = get_module(pruned_model, layer_name + '.2'), get_module(last_submodel, layer_name + '.2')
                    cur_fw.w.data[0, cur_filter_i] = last_fw.w.data[0, last_filter_i]
                
                if w_factor > 0:
                    cur_filter_i += 1
                if last_w_factor > 0:
                    last_filter_i += 1
        
        return pruned_model, pruning_info
    
    def absorb_sub_model(self, fbs_model: nn.Module, sub_model: nn.Module, pruning_info: dict, alpha=1.):
        if alpha == 0.:
            return
        for layer_name, feature_boosting_w in pruning_info.items():
            unpruned_filters_index = feature_boosting_w.nonzero(as_tuple=True)[0]
            
            fbs_layer = get_module(fbs_model, layer_name)
            sub_model_layer = get_module(sub_model, layer_name)

            for fi_in_sub_layer, fi_in_fbs_layer in enumerate(unpruned_filters_index):
                fbs_layer.raw_conv2d.weight.data[fi_in_fbs_layer] = (1. - alpha) * fbs_layer.raw_conv2d.weight.data[fi_in_fbs_layer] + \
                    alpha * sub_model_layer[0].weight.data[fi_in_sub_layer]
                for k in ['weight', 'bias', 'running_mean', 'running_var']:
                    getattr(fbs_layer.bn, k).data[fi_in_fbs_layer] = (1. - alpha) * getattr(fbs_layer.bn, k).data[fi_in_fbs_layer] + \
                        alpha * getattr(sub_model_layer[1], k).data[fi_in_sub_layer]
    
    
class DAFBSSubModelExtractor(FBSSubModelExtractor):
    def __init__(self) -> None:
        super().__init__()
        # self.debug_sample_i = 0
        # self.last_final_ws = None
        
    @abstractmethod
    def generate_pruning_strategy(self, fbs_model: nn.Module, samples: torch.Tensor):
        with torch.no_grad():
            fbs_model.eval()
            self.cur_output = fbs_model(samples)
    
    @abstractmethod
    def get_final_w(self, fbs_model: nn.Module, samples: torch.Tensor, layer_name: str, w: torch.Tensor):
        # import matplotlib.pyplot as plt
        # plt.imshow(w.cpu().numpy(), cmap='Greys')
        # # plt.colorbar()
        # plt.xlabel('Filters')
        # plt.ylabel('Samples')
        # plt.tight_layout()
        # plt.savefig(os.path.join(res_save_dir, f'{layer_name}.png'), dpi=300)
        # plt.clf()
        # w_sum = w.sum(0)
        # w_argsort = w_sum.argsort(descending=True)
        # return w[self.debug_sample_i]
        # x = self.cur_output
        # each_sample_entropy = -(x.softmax(1) * x.log_softmax(1)).sum(1)
        
        # hardest_sample_index = w.sum(1).argmax()
        # return w[hardest_sample_index]
        # [0.0828, 0.1017, 0.0575, 0.3081, 0.1511, 0.3634, 0.3388, 0.3942, 0.2475, 0.3371, 0.5837, 0.145, 0.4428, 0.2159, 0.4028] 0.27815999999999996

        x = self.cur_output
        each_sample_entropy = -(x.logits.softmax(1) * x.logits.log_softmax(1)).sum(1)
        hardest_sample_index = each_sample_entropy.argmax()
        res = w[hardest_sample_index]
        return res
        
        # if self.last_final_ws is not None:
        #     intersection_ratio = (self.last_final_w == res).sum() / (res > 0).sum()
        #     print('intersection ratio: ', intersection_ratio)
        
        # self.last_final_ws[layer_name] = res
        
        
        
        # indices = (-w).sum(0).topk((w[0] == 0).sum())[1]
        # boosting = w.max(0)[0]
        # boosting[indices] = 0.
        # return boosting
        
        # return w[0]
        
        
def tent_as_detector(model, x, num_iters=1, lr=1e-4, l1_wd=0., strategy='ours'):
    model = deepcopy(model)
    before_model = deepcopy(model)
    
    from methods.tent import tent
    
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, weight_decay=l1_wd)
    from models.resnet_cifar.model_manager import ResNetCIFARManager
    tented_model = tent.Tent(model, optimizer, ResNetCIFARManager, steps=num_iters)
    
    tent.configure_model(model)
    tented_model(x)
    
    filters_sen_info = {}
    
    last_conv_name = None
    for (name, m1), m2 in zip(model.named_modules(), before_model.modules()):
        if isinstance(m1, nn.Conv2d):
            last_conv_name = name
            
        if not isinstance(m1, nn.BatchNorm2d):
            continue
        
        with torch.no_grad():
            features_weight_diff = ((m1.weight.data - m2.weight.data).abs())
            features_bias_diff = ((m1.bias.data - m2.bias.data).abs())
            
            features_diff = features_weight_diff + features_bias_diff
            
            features_diff_order = features_diff.argsort(descending=False)
            
            if strategy == 'ours':
                untrained_filters_index = features_diff_order[: int(len(features_diff) * 0.8)]
            elif strategy == 'random':
                untrained_filters_index = torch.randperm(len(features_diff))[: int(len(features_diff) * 0.8)]
            elif strategy == 'inversed_ours':
                untrained_filters_index = features_diff_order.flip(0)[: int(len(features_diff) * 0.8)]
            elif strategy == 'none':
                untrained_filters_index = None
            
            filters_sen_info[name] = dict(untrained_filters_index=untrained_filters_index, conv_name=last_conv_name)
            
    return filters_sen_info


class SGDF(torch.optim.SGD):

    @torch.no_grad()
    def step(self, model, conv_filters_sen_info, filters_sen_info, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            # assert len([i for i in model.named_parameters()]) == len([j for j in group['params']])

            for (name, _), p in zip(model.named_parameters(), group['params']):
                if p.grad is None:
                    continue
                
                layer_name = '.'.join(name.split('.')[0:-1])
                if layer_name in filters_sen_info.keys():
                    untrained_filters_index = filters_sen_info[layer_name]['untrained_filters_index']
                elif layer_name in conv_filters_sen_info.keys():
                    untrained_filters_index = conv_filters_sen_info[layer_name]['untrained_filters_index']
                else:
                    untrained_filters_index = []
                
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                d_p[untrained_filters_index] = 0.
                p.add_(d_p, alpha=-group['lr'])

        return loss

    
if __name__ == '__main__':
    set_random_seed(0)
    
    import sys
    tag = sys.argv[1]
    # alpha = 0.4
    alpha = 0.2
    # alpha = float(sys.argv[1])
    
    fbs_model_path = sys.argv[1]

    cur_time_str = get_cur_time_str()
    res_save_dir = f'logs/experiments_trial/CIFAR100C/ours_fbs_more_challenging/{cur_time_str[0:8]}/{cur_time_str[8:]}-{tag}'
    os.makedirs(res_save_dir)
    
    import shutil
    shutil.copytree(os.path.dirname(__file__),
                    os.path.join(res_save_dir, 'method'), ignore=shutil.ignore_patterns('*.pt', '*.pth', 'log', '__pycache__'))
    logger.info(f'res save dir: {res_save_dir}')
    
    # model = torch.load('logs/experiments_trial/CIFAR100C/ours_dynamic_filters/20220801/152138-0.6_l1wd=1e-8/best_model_0.80.pt')
    # model = torch.load('logs/experiments_trial/CIFAR100C/ours_dynamic_filters/20220801/232913-sample_subnetwork/best_model_0.80.pt')
    model = torch.load(fbs_model_path)
    
    # model = torch.load('logs/experiments_trial/CIFAR100C/ours_dynamic_filters/20220729/002444-0.4/best_model_0.40.pt')
    
    # import sys
    # sys.path.append('/data/xgf/legodnn_and_domain_adaptation')
    xgf_model = torch.load('logs/experiments_trial/CIFAR100C/ours_dynamic_filters/20220731/224212-cifar10_svhn_raw/last_model.pt')
    # xgf_model = torch.load('/data/xgf/legodnn_and_domain_adaptation/results_scaling_da/image_classification/CIFAR100C_resnet18/onda/offline_l1/s4/20220607/204211/last_model.pt')
    # test_dataloader = get_source_dataloader('CIFAR100', 256, 4, 'test', False, False, False)
    # test_dataloader = get_target_dataloaders('CIFAR100C', [7], 128, 4, 'test', False, False, False)[0] # snow, xgf 0.3914
    # test_dataloaders = get_target_dataloaders('CIFAR100C', list(range(15)), 128, 4, 'test', False, False, False) # defocus_blur, xgf 0.2836
    # test_dataloaders = get_target_dataloaders('RotatedCIFAR100', list(range(18)), 128, 4, 'test', False, False, False)
    train_dataloaders = [
        get_source_dataloader(dataset_name, 128, 4, 'train', True, None, True) for dataset_name in ['SVHN', 'CIFAR10', 'SVHN']
    ][::-1] * 10
    test_dataloaders = [
        get_source_dataloader('USPS', 128, 4, 'test', False, False, False),
        get_source_dataloader('STL10-wo-monkey', 128, 4, 'test', False, False, False),
        get_source_dataloader('MNIST', 128, 4, 'test', False, False, False),
    ][::-1] * 10
    y_offsets = [10, 0, 10][::-1] * 10
    domain_names = ['USPS', 'STL10', 'MNIST'][::-1] * 10
    # train_dataloader = get_source_dataloader('CIFAR100', 128, 4, 'train', True, None, True)
    # acc = ResNetCIFARManager.get_accuracy(model, test_dataloader, 'cuda')
    # print(acc)
    # baseline_accs = [0.1012, 0.1156, 0.0529, 0.2836, 0.1731, 0.3765, 0.3445, 0.3914, 0.2672, 0.3289, 0.5991, 0.1486, 0.4519, 0.1907, 0.3929]
    # accs = []
    
    baseline_before, baseline_after, ours_before, ours_after = [], [], [], []
    last_pruned_model, last_pruning_info = None, None
    # y_offset = 0
    for ti, (test_dataloader, y_offset) in enumerate(zip(test_dataloaders, y_offsets)):
        samples, labels = next(iter(test_dataloader))
        samples, labels = samples.cuda(), labels.cuda()
        labels += y_offset
        
        def bn_cal(_model: nn.Module):
            for n, m in _model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
                    m.training = True
                    m.train()
            for _ in range(100): # ~one epoch
                x, y = next(train_dataloaders[ti])
                x = x.cuda()
                _model(samples)
        
        def shot(_model: nn.Module, lr=6e-4, num_iters_scale=1, wd=0.):
            # print([n for n, p in model.named_parameters()])
            _model.requires_grad_(True)
            _model.linear.requires_grad_(False)
            import torch.optim
            optimizer = torch.optim.SGD([p for p in _model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=wd)
            device = 'cuda'
            
            for _ in range(100 * num_iters_scale):
                x = samples
                _model.train()
                output = ResNetCIFARManager.forward(_model, x)
                
                def Entropy(input_):
                    entropy = -input_ * torch.log(input_ + 1e-5)
                    entropy = torch.sum(entropy, dim=1)
                    return entropy 
                
                softmax_out = nn.Softmax(dim=1)(output)
                entropy_loss = torch.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                loss = entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        def shot_w_part_filter(_model: nn.Module, lr=6e-4, num_iters_scale=1, wd=0.):
            # print([n for n, p in model.named_parameters()])
            _model.requires_grad_(True)
            _model.linear.requires_grad_(False)
            import torch.optim
            optimizer = SGDF([p for p in _model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=wd)
            device = 'cuda'
            
            filters_sen_info = tent_as_detector(_model, samples, strategy='ours')
            conv_filters_sen_info = {v['conv_name']: v for _, v in filters_sen_info.items()}
            
            for _ in range(100 * num_iters_scale):
                x = samples
                _model.train()
                output = ResNetCIFARManager.forward(_model, x)
                
                def Entropy(input_):
                    entropy = -input_ * torch.log(input_ + 1e-5)
                    entropy = torch.sum(entropy, dim=1)
                    return entropy 
                
                softmax_out = nn.Softmax(dim=1)(output)
                entropy_loss = torch.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                loss = entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(_model, conv_filters_sen_info, filters_sen_info)
                
        def tent(_model: nn.Module):
            from methods.tent import tent
            _model = tent.configure_model(_model)
            params, param_names = tent.collect_params(_model)
            optimizer = torch.optim.Adam(params, lr=1e-4)
            tent_model = tent.Tent(_model, optimizer, ResNetCIFARManager, steps=1)
            
            tent.configure_model(_model)
            tent_model(samples)
            
        def tent_configure_bn(_model):
            """Configure model for use with tent."""
            # train mode, because tent optimizes the model to minimize entropy
            # _model.train()
            # # disable grad, to (re-)enable only what tent updates
            # _model.requires_grad_(False)
            # configure norm for tent updates: enable grad + force batch statisics
            for m in _model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    
                    # m.track_running_stats = True
                    # m.momentum = 1.0
                    
                # # FIXME
                # from methods.ours_dynamic_filters.extract_submodel import FeatureBoosting
                # # if isinstance(m, FeatureBoosting):
                # if m.__class__.__name__ == 'FeatureBoosting':
                #     m.requires_grad_(True)
                    
            return model
            
        def sl(_model: nn.Module, lr=6e-4, num_iters_scale=1, wd=0.):
            _model.requires_grad_(True)
            _model.linear.requires_grad_(False)
            import torch.optim
            optimizer = torch.optim.SGD([p for p in _model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=wd)
            device = 'cuda'
            
            for _ in range(100 * num_iters_scale):
                x = samples
                _model.train()
                loss = ResNetCIFARManager.forward_to_gen_loss(_model, x, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        model_extractor = DAFBSSubModelExtractor()
        model1 = model_extractor.extract_submodel_via_a_sample(model,samples[0])
        pruned_model, pruning_info = model_extractor.extract_submodel_via_samples_and_last_submodel(model, samples, None, None)
        # print(pruned_model)
        # print(get_model_size(pruned_model, True))
        # bn_cal(pruned_model)
        acc = ResNetCIFARManager.get_accuracy(pruned_model, test_dataloader, 'cuda', y_offset)
        print(acc)
        ours_before += [acc]
        # tent(pruned_model)
        # bn_cal(pruned_model)
        shot_w_part_filter(pruned_model, 6e-4, 1, 1e-3)
        # sl(pruned_model)
        acc = ResNetCIFARManager.get_accuracy(pruned_model, test_dataloader, 'cuda', y_offset)
        print(acc)
        ours_after += [acc]
        
        last_pruned_model, last_pruning_info = deepcopy(pruned_model), deepcopy(pruning_info)
        model_extractor.absorb_sub_model(model, pruned_model, pruning_info, alpha)
        
        # xgf_model = torch.load('/data/xgf/legodnn_and_domain_adaptation/results_scaling_da/image_classification/CIFAR100C_resnet18/onda/offline_l1/s8/20220607/212448/last_model.pt')
        # xgf_model = torch.load('/data/xgf/legodnn_and_domain_adaptation/results_scaling_da/image_classification/CIFAR100C_resnet18/onda/offline_l1/s4/20220607/204211/last_model.pt')

        # print(xgf_model)
        # acc = ResNetCIFARManager.get_accuracy(xgf_model, test_dataloader, 'cuda', y_offset)
        # print(acc)
        # baseline_before += [acc]
        # # tent(xgf_model)
        # shot(xgf_model)
        # # sl(xgf_model)
        # acc = ResNetCIFARManager.get_accuracy(xgf_model, test_dataloader, 'cuda', y_offset)
        # print(acc)
        # baseline_after += [acc]
        # print()
    #     diff = acc - baseline_accs[ti]
    #     print(f'domain {ti}, model size {get_model_size(pruned_model, True):.3f}MB, diff: {diff:.4f}')
    # print(accs, sum(accs) / len(accs))
    
    import matplotlib.pyplot as plt
    from visualize.util import *
    set_figure_settings(3)
    
    def avg(arr):
        return sum(arr) / len(arr)

    # plt.plot(list(range(len(test_dataloaders))), baseline_before, lw=2, linestyle='--', color=BLUE, label=f'L1 before DA ({avg(baseline_before):.4f})')
    # plt.plot(list(range(len(test_dataloaders))), baseline_after, lw=2, linestyle='-', color=BLUE, label=f'L1 after DA ({avg(baseline_after):.4f})')
    plt.plot(list(range(len(test_dataloaders))), ours_before, lw=2, linestyle='--', color=RED, label=f'ours before DA ({avg(ours_before):.4f})')
    plt.plot(list(range(len(test_dataloaders))), ours_after, lw=2, linestyle='-', color=RED, label=f'ours after DA ({avg(ours_after):.4f})')
    plt.xlabel('domains')
    plt.ylabel('accuracy')
    plt.xticks(list(range(len(domain_names))), domain_names, rotation=90)
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(res_save_dir, 'main.png'), dpi=300)
    plt.clf()
    
    torch.save((baseline_before, baseline_after, ours_before, ours_after), os.path.join(res_save_dir, 'main.png.data'))
    
    # with open('./tmp.csv', 'a') as f:
    #     f.write(f'{alpha:.2f},{avg(baseline_after):.4f},{avg(ours_after):.4f}')
    
    # std: logs/experiments_trial/CIFAR100C/ours_dynamic_filters/20220730/161404-submodel/main.png
    
    
    # accs = []
    # for i in tqdm.tqdm(range(100)):
    #     model_extractor.debug_sample_i = i
    #     pruned_model = model_extractor.extract_submodel_via_samples(model, samples)
    #     acc = ResNetCIFARManager.get_accuracy(pruned_model, test_dataloader, 'cuda')
    #     accs += [acc]
    
    # import matplotlib.pyplot as plt
    # plt.plot(list(range(100)), accs)
    # plt.savefig('./tmp.png', dpi=300)
    # plt.clf()
    
    
    # ------------------------------
    # perf test
    
    # sample, _ = next(iter(test_dataloader))
    # sample = sample[0: 1].cuda()
    
    # pruned_model = FBSSubModelExtractor().extract_submodel_via_a_sample(model, sample)
    
    # bs = 1
    # def perf_test(model, batch_size, device):
    #     model = model.to(device)
    #     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        
    #     # warmup
    #     for _ in range(100):
    #         rand_input = torch.rand((batch_size, 3, 32, 32)).to(device)
    #         o = model(rand_input)
        
    #     forward_latency = 0.
    #     backward_latency = 0.

    #     for _ in range(100):
    #         rand_input = torch.rand((batch_size, 3, 32, 32)).to(device)
    #         optimizer.zero_grad()
            
    #         s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #         s.record()
    #         o = model(rand_input)
    #         e.record()
    #         torch.cuda.synchronize()
    #         forward_latency += s.elapsed_time(e) / 1000.
            
    #         loss = ((o - 1) ** 2).sum()
            
    #         s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #         s.record()
    #         loss.backward()
    #         optimizer.step()
    #         e.record()
    #         torch.cuda.synchronize()
    #         backward_latency += s.elapsed_time(e) / 1000.
            
    #     forward_latency /= 100
    #     backward_latency /= 100
        
    #     print(forward_latency, backward_latency)
    
    # for bs in [1, 128]:
    #     for device in ['cuda', 'cpu']:
    #         for m in [model, pruned_model]:
    #             print(bs, device)
    #             perf_test(m, bs, device)