from turtle import forward
from typing import Optional
import torch
import copy
from torch import nn
#from methods.utils.data import get_source_dataloader
from utils.dl.common.model import get_model_device, get_model_latency, get_model_size, get_module, get_super_module, set_module
from utils.common.log import logger


"""
No real speedup.
But it's ok because our big model just forward for one time to find the best sub-model.
The sub-model doesn't contain filter selection modules. It's just a normal model.
"""

class KTakesAll(nn.Module):
    def __init__(self, k):
        super(KTakesAll, self).__init__()

        self.k = k
        
    def forward(self, g: torch.Tensor):
        # if self.k == 0.:
        #     t = g
        #     t = t / torch.sum(t, dim=1).unsqueeze(1) * t.size(1)
        #     return t.unsqueeze(2).unsqueeze(3)
        #     t = g
        #     t = t / torch.sum(t, dim=1).unsqueeze(1) * t.size(1)
        #     # print('000', t.size())
        #     t = t.unsqueeze(2).unsqueeze(3).mean((0, 2, 3)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #     # print('111', t.size())
        #     # print(t)
        #     return t
        # # assert x.dim() == 2
        # print(g)
        k = int(g.size(1) * self.k)
        
        i = (-g).topk(k, 1)[1]
        t = g.scatter(1, i, 0)
        # t = t / torch.sum(t, dim=1).unsqueeze(1) * t.size(1)
        # print(t)
        
        return t.unsqueeze(2).unsqueeze(3)
        # g = g.mean(0).unsqueeze(0)
        
        # k = int(g.size(1) * self.k)
        
        # i = (-g).topk(k, 1)[1]
        # t = g.scatter(1, i, 0)
        # t = t / torch.sum(t, dim=1).unsqueeze(1) * t.size(1)
        
        # return t.unsqueeze(2).unsqueeze(3)

# class NoiseAdd(nn.Module):
#     def __init__(self):
#         super(NoiseAdd, self).__init__()

#         self.training = True
        
#     def forward(self, x):
#         if self.training:
#             return x + torch.randn_like(x, device=x.device)
#         else:
#             return x

class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()
        
    def forward(self, x):
        return x.abs()


class DomainDynamicConv2d(nn.Module):
    def __init__(self, raw_conv2d: nn.Conv2d, raw_bn: nn.BatchNorm2d, k: float, bn_after_fc=False):
        super(DomainDynamicConv2d, self).__init__()

        assert not bn_after_fc
        
        self.filter_selection_module = nn.Sequential(
            Abs(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(raw_conv2d.in_channels, raw_conv2d.out_channels),
            # nn.Conv2d(raw_conv2d.in_channels, raw_conv2d.out_channels // 16, kernel_size=1, bias=False),
            
            # nn.Linear(raw_conv2d.in_channels, raw_conv2d.out_channels // 16),
            # nn.BatchNorm1d(raw_conv2d.out_channels // 16) if bn_after_fc else nn.Identity(),
            # nn.ReLU(),
            # nn.Linear(raw_conv2d.out_channels // 16, raw_conv2d.out_channels),
        
            # nn.BatchNorm1d(raw_conv2d.out_channels),
            nn.ReLU(),
            # NoiseAdd(),
            # nn.Sigmoid()
            # L1RegTrack(),
            # KTakesAll(k)
        )
        self.k_takes_all = KTakesAll(k)
        
        self.raw_conv2d = raw_conv2d
        self.bn = raw_bn # remember clear the original BNs in the network
        
        nn.init.constant_(self.filter_selection_module[3].bias, 1.)
        nn.init.kaiming_normal_(self.filter_selection_module[3].weight)
        
        self.cached_raw_w = None
        self.l1_reg_of_raw_w = None
        self.cached_w = None
        self.static_w = None
        self.pruning_ratios = None

        
    def forward(self, x):
        raw_x = self.bn(self.raw_conv2d(x))
        
        # if self.k_takes_all.k < 1e-7:
        #     return raw_x
        
        if self.static_w is None:
            raw_w = self.filter_selection_module(x)
            
            self.cached_raw_w = raw_w
            # self.l1_reg_of_raw_w = raw_w.norm(1, dim=1).mean()
            self.l1_reg_of_raw_w = raw_w.norm(1)
            
            w = self.k_takes_all(raw_w)
            
            # w = w.unsqueeze(2).unsqueeze(3)
            
            # if self.training:
            #     soft_w = torch.max(torch.zeros_like(raw_w), torch.min(torch.ones_like(raw_w), 
            #                                                         1.2 * (torch.sigmoid(raw_w + torch.randn_like(raw_w))) - 0.1))
            # else:
            #     soft_w = torch.max(torch.zeros_like(raw_w), torch.min(torch.ones_like(raw_w), 
            #                                                         1.2 * (torch.sigmoid(raw_w)) - 0.1))
            
            # w = soft_w.detach().clone()
            # w[w < 0.5] = 0.
            # w[w >= 0.5] = 1.
            # w = w + soft_w - soft_w.detach()
            
            # w = w.unsqueeze(2).unsqueeze(3)
            # soft_w = soft_w.unsqueeze(2).unsqueeze(3)
            # self.l1_reg_of_raw_w = soft_w.norm(1)
            
            self.cached_w = w
            
            # print(w.size(), x.size(), raw_x.size())
        else:
            w = self.static_w.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            
        if self.pruning_ratios is not None:
            # self.pruning_ratios += [1. - float((w_of_a_asample > 0.).sum() / w_of_a_asample.numel()) for w_of_a_asample in w]
            self.pruning_ratios += [torch.sum(w > 0.) / w.numel()]
        
        return raw_x * w
    
    # def to_static(self):
    #     global_w = self.cached_raw_w.detach().topk(0.25, 1)[0].mean(0).unsqueeze(0)
    #     global_w = self.k_takes_all(global_w).squeeze(0)
    #     self.static_w = global_w
        
    # def to_dynamic(self):
    #     self.static_w = None
        

def boost_raw_model_with_filter_selection(model: nn.Module, init_k: float, bn_after_fc=False, ignore_layers=None, perf_test=True, model_input_size: Optional[tuple]=None):
    model = copy.deepcopy(model)

    device = get_model_device(model)
    if perf_test:
        before_model_size = get_model_size(model, True)
        before_model_latency = get_model_latency(
            model, model_input_size, 50, device, 50)

    # clear original BNs
    num_original_bns = 0
    last_conv_name = None
    conv_bn_map = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name
        if isinstance(module, nn.BatchNorm2d) and (ignore_layers is not None and last_conv_name not in ignore_layers):
            # set_module(model, name, nn.Identity())
            num_original_bns += 1
            conv_bn_map[last_conv_name] = name
    
    num_conv = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and (ignore_layers is not None and name not in ignore_layers):
            set_module(model, name, DomainDynamicConv2d(module, get_module(model, conv_bn_map[name]), init_k, bn_after_fc))
            num_conv += 1
            
    assert num_conv == num_original_bns
    
    for bn_layer in conv_bn_map.values():
        set_module(model, bn_layer, nn.Identity())

    if perf_test:
        after_model_size = get_model_size(model, True)
        after_model_latency = get_model_latency(
            model, model_input_size, 50, device, 50)

        logger.info(f'raw model -> raw model w/ filter selection:\n'
                    f'model size: {before_model_size:.3f}MB -> {after_model_size:.3f}MB '
                    f'latency: {before_model_latency:.6f}s -> {after_model_latency:.6f}s')
        
    return model, conv_bn_map


def get_l1_reg_in_model(boosted_model):
    res = 0.
    for name, module in boosted_model.named_modules():
        if isinstance(module, DomainDynamicConv2d):
            res += module.l1_reg_of_raw_w
    return res

            
def get_cached_w(model):
    res = []
    for name, module in model.named_modules():
        if isinstance(module, DomainDynamicConv2d):
            res += [module.cached_w]
    return torch.cat(res, dim=1)


def set_pruning_rate(model, k):
    for name, module in model.named_modules():
        if isinstance(module, KTakesAll):
            module.k = k


def get_cached_raw_w(model):
    res = []
    for name, module in model.named_modules():
        if isinstance(module, DomainDynamicConv2d):
            res += [module.cached_raw_w]
    return torch.cat(res, dim=1)


def start_accmu_flops(model):
    for name, module in model.named_modules():
        if isinstance(module, DomainDynamicConv2d):
            module.pruning_ratios = []
            

def get_accmu_flops(model):
    layer_res = {}
    total_res = []
    
    for name, module in model.named_modules():
        if isinstance(module, DomainDynamicConv2d):
            layer_res[name] = module.pruning_ratios
            total_res += module.pruning_ratios
            module.pruning_ratios = None
            
    avg_pruning_ratio = sum(total_res) / len(total_res)
    return layer_res, total_res, avg_pruning_ratio


def convert_boosted_model_to_static(boosted_model, a_few_data):
    boosted_model(a_few_data)

    for name, module in boosted_model.named_modules():
        if isinstance(module, DomainDynamicConv2d):
            module.to_static()
            # TODO: use fn3 techniques
            
            
def ensure_boosted_model_to_dynamic(boosted_model):
    for name, module in boosted_model.named_modules():
        if isinstance(module, DomainDynamicConv2d):
            module.to_dynamic()
            
            
def train_only_gate(model):
    gate_params = []
    for n, p in model.named_parameters():
        if 'filter_selection_module' in n:
            gate_params += [p]
        else:
            p.requires_grad = False
    return gate_params
    
if __name__ == '__main__':
    # rand_input = torch.rand((256, 3, 32, 32))
    # conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    # new_conv = DomainDynamicConv2d(conv, 0.1)
    
    # train_dataloader = get_source_dataloader('CIFAR100', 256, 4, 'train', True, None, True)
    # rand_input, _ = next(train_dataloader)
    
    # start_accmu_flops(new_conv)

    # new_conv(rand_input)

    # _, total_pruning_ratio, avg_pruning_ratio = get_accmu_flops(new_conv)

    # import matplotlib.pyplot as plt
    # plt.hist(total_pruning_ratio)
    # plt.savefig('./tmp.png')
    # plt.clf()
    
    # print(avg_pruning_ratio)

    

    # with torch.no_grad():
    #     conv(rand_input)
    #     new_conv(rand_input)

    # from torchvision.models import resnet18
    
    # model = resnet18()
    # boost_raw_model_with_filter_selection(model, 0.5, True, (1, 3, 224, 224))
    
    # rand_input = torch.rand((2, 3, 32, 32))
    # conv = nn.Conv2d(3, 4, 3, 1, 1, bias=False)
    # w = torch.rand((1, 4)).repeat(2, 1)
    
    # with torch.no_grad():
    #     o1 = conv(rand_input) * w.unsqueeze(2).unsqueeze(3)
    #     print(w)
        
    #     w = w.mean(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    #     print(w)
    #     conv.weight.data.mul_(w)
        
    #     o2 = conv(rand_input)

    #     diff = ((o1 - o2) ** 2).sum()
    #     print(diff)
    
    
    # rand_input = torch.rand((2, 3, 32, 32))
    # conv1 = nn.Conv2d(3, 6, 3, 1, 1, bias=False)
    # conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=False, groups=3)
    
    # print(conv1.weight.data.size(), conv2.weight.data.size())
    
    # import time
    # import torch
    # from utils.dl.common.model import get_model_latency
    
    # # s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # # s.record()
    # # # TODO
    # # e.record()
    # # torch.cuda.synchronize()
    # # time_usage = s.elapsed_time(e) / 1000.
    # # print(time_usage)
    
    # data = [torch.rand((512, 3, 3)).cuda() for _ in range(512)]
    # # t1 = time.time()
    # # for i in range(300): d = torch.stack(data)  
    # # t2 = time.time()
    # # for i in range(300): d = torch.cat(data).view(512, 512, 3, 3) 
    # # t3 = time.time()
    # # print("torch.stack time: {}, torch.cat time: {}".format(t2 - t1, t3 - t2))
    
    # s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # s.record()
    # for i in range(300): d = torch.stack(data)  
    # e.record()
    # torch.cuda.synchronize()
    # time_usage = s.elapsed_time(e) / 1000.
    # print(time_usage)
    
    # s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # s.record()
    # for i in range(300): d = torch.cat(data).view(512, 512, 3, 3) 
    # e.record()
    # torch.cuda.synchronize()
    # time_usage = s.elapsed_time(e) / 1000.
    # print(time_usage)
    
    
    # from models.resnet_cifar.resnet_cifar_3 import resnet18
    # model = resnet18()
    
    # full_l1_reg = 0.
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         w = torch.ones((256, module.out_channels))
    #         w[:, (module.out_channels // 2):] = 0.
    #         full_l1_reg += w.norm(1)
    
    # full_l1_reg /= 2
        
    # print(f'{full_l1_reg:.3e}')
    
    # def f(x):
    #     # x = x - 0.5
    #     return torch.max(torch.zeros_like(x), torch.min(torch.ones_like(x), 1.2 * torch.sigmoid(x) - 0.1))
    
    # x = torch.arange(-2, 2, 0.01).float()
    # y = f(x)
    
    # print(f(torch.FloatTensor([0.])))
    # print(f(torch.FloatTensor([0.5])))
    
    # import matplotlib.pyplot as plt
    
    # plt.plot(x, y)
    # plt.savefig('./tmp.png')
    
    # rand_input = torch.rand((256, 3, 32, 32))
    # conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    # new_conv = DomainDynamicConv2d(conv, 0.1)
    
    # new_conv(rand_input)
    
    # conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    # new_conv = DomainDynamicConv2d(conv, nn.BatchNorm2d(64), 0.1)
    # print(new_conv.filter_selection_module[5].training)
    # new_conv.eval()
    # print(new_conv.filter_selection_module[5].training)
    
    n = KTakesAll(0.6)

    rand_input = torch.rand((1, 5))
    print(n(rand_input))