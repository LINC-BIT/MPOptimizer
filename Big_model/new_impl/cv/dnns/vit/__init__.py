import timm
from timm.models._factory import load_checkpoint
import torch
import os
from torch import nn 
from torch.jit import Final
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.dl.common.model import get_model_device, set_module
import torch.nn.functional as F
from utils.common.log import logger


# class SoftmaxIgnoringZero(nn.Module):
#     def __init__(self):
#         super(SoftmaxIgnoringZero, self).__init__()
    
#     def forward(self, x: torch.Tensor):
#         # non_zero_x_indexes = x.nonzero(as_tuple=True)[0]
#         # non_zero_x = x[non_zero_x_indexes]
#         # non_zero_x_softmax = F.softmax(non_zero_x, self.dim, _stacklevel=5)
#         # res = torch.zeros_like(x)

#         # original: e^i / \sum_i e^i
#         # ignoring zero: e^i
#         # print(x)
        
#         non_zero_mask = x != 0
        
#         if non_zero_mask.sum() == x.numel():
#             return F.softmax(x, -1)
        
#         t = non_zero_mask.sum(-1)
#         assert t.view(-1).unique().size(0) == 1, f'{t.view(-1).unique()}, {x.size()}' # all vectors in the softmaxed dim has the same number of 0
#         # assert t.view(-1).unique().size(0) <= 2, f'{t.view(-1).unique()}, {x.size()}' # all vectors in the softmaxed dim has the same number of 0 or has no 0
#         non_zero_x = torch.masked_select(x, non_zero_mask)
        
#         non_zero_x = non_zero_x.view(*(list(x.size())[0: -1] + [t.view(-1)[0].item()]))
        
#         # print(non_zero_x)
        
#         non_zero_x_softmax = F.softmax(non_zero_x, -1)
        
#         a = x.nonzero(as_tuple=True)[-1]
#         a = a.view(*non_zero_x_softmax.size())
#         x = x.scatter(x.dim() - 1, a, non_zero_x_softmax)
        
#         return x


class SoftmaxIgnoringZero(nn.Module):
    def __init__(self):
        super(SoftmaxIgnoringZero, self).__init__()
    
    def f(self, x):
        # return x / (x + 1e-8)
        return 1.
    
    def forward(self, x: torch.Tensor):
        res = F.softmax(x, -1)
        return res * self.f(x)


class PrunableAttention(nn.Module):
    """
    https://github.com/lucidrains/vit-pytorch
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qkv_bias = False):
        super().__init__()
        self.inner_dim = inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.num_heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim * 3, bias = qkv_bias)

        # self.proj = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()
        
        self.proj = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # qkv = self.qkv(x).chunk(3, dim = -1)
        raw_qkv = self.qkv(x)
        
        self.inner_dim = (raw_qkv.size(-1) - self.proj.in_features) // 2
        qkv = raw_qkv[:, :, 0: self.inner_dim], raw_qkv[:, :, self.inner_dim: self.inner_dim * 2], raw_qkv[:, :, self.inner_dim * 2:]
        
        # print('v', qkv[0].size(), qkv[0].sum((0, 1))[0: 10], qkv[0].sum((0, 1)).nonzero(as_tuple=True)[0].size())
        
        # raw_v = qkv[2]
        # print('after_fbs_q, after_fbs_k', qkv[0].sum((0, 1))[0: 10], qkv[0].sum((0, 1)).nonzero(as_tuple=True)[0].size(),
        #       qkv[1].sum((0, 1))[0: 10], qkv[1].sum((0, 1)).nonzero(as_tuple=True)[0].size(),)
        # print('after_fbs_v', raw_v.size(), raw_v.sum((0, 1))[0: 10], raw_v.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        # print('q, before rearrage', qkv[0].size())
        q, k, v = qkv
        # print('raw qkv size', q.size(), k.size(), v.size())
        # exit()
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        # print('raw qkv size', q.size(), k.size(), v.size())
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # print('q, k, dots, after rearrage', q.size(), k.transpose(-1, -2).size(), dots.size())
        
        attn = self.attend(dots)
        # attn = dots
        attn = self.dropout(attn)

        # print(attn)
        # print('attn', attn.size(), attn.sum((0, 1))[0: 10], attn.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        # print('attn', attn.size(), attn.sum((0, 1))[0: 10], attn.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        # print('v2', v.size())
        out = torch.matmul(attn, v)
        # print('out1', out.size())
        # NOTE: just for trial debug
        # out = v
        
        # print('out before rerange', out.size())
        
        # print(v.size(), v)
        # exit()
        
        out = rearrange(out, 'b h n d -> b n (h d)')

        # print('out', out.size(), out.sum((0, 1))[0: 10], out.sum((0, 1)).nonzero(as_tuple=True)[0].size())
        # exit()
        
        res = self.proj_dropout(self.proj(out))
        
        # res = self.proj_dropout(
        #     F.linear(self.proj.weight.T, out.T, self.proj.bias)
        # )
        # print(self.proj, self.proj_dropout)
        # print('res', res.size(), res.sum((0, 1))[0: 10], res.sum((0, 1)).nonzero(as_tuple=True)[0].size())

        return res
    

def make_attention_prunable(vit):
    for block in vit.blocks:
        attn = block.attn
        
        assert attn.attn_drop.p == attn.proj_drop.p

        prunable_attn = PrunableAttention(
            dim=attn.head_dim * attn.num_heads,
            heads=attn.num_heads,
            dim_head=attn.head_dim,
            dropout=attn.attn_drop.p,
            qkv_bias=attn.qkv.bias is not None
        )
        prunable_attn.qkv.weight.copy_(attn.qkv.weight)
        if attn.qkv.bias is not None:
            prunable_attn.qkv.bias.copy_(attn.qkv.bias)
        prunable_attn.proj.weight.copy_(attn.proj.weight)
        prunable_attn.proj.bias.copy_(attn.proj.bias)
        
        set_module(block, 'attn', prunable_attn)
        
        
@torch.no_grad()
def vit_l_16(pretrained=True, num_classes=None) -> nn.Module:
    # https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k
    res = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k',
                            num_classes=num_classes)
        
    if pretrained:
        checkpoint_path = os.path.join(os.path.dirname(__file__), 
                                       'weights/vit_large_patch16_224.augreg_in21k_ft_in1k.bin')
        def filter_fn(state_dict, _):
            if num_classes is None: # use fine-tuned in1k fc head
                return state_dict
            else: # use a new linear
                del state_dict['head.weight']
                del state_dict['head.bias']
                return state_dict
            
        load_checkpoint(res, checkpoint_path, strict=False, filter_fn=filter_fn)
    
    res.eval()
    input_sample = torch.rand(2, 3, 224, 224)
    o1 = res(input_sample)
    
    make_attention_prunable(res)
    res.eval()
    o2 = res(input_sample)
    
    assert ((o1 - o2) ** 2).sum() < 1e-5
    return res


from timm.models.vision_transformer import VisionTransformer

@torch.no_grad()
def vit_b_16(pretrained=True, num_classes=None) -> VisionTransformer:
    # https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k
    res = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k',
                            num_classes=num_classes)
        
    if pretrained:
        checkpoint_path = os.path.join(os.path.dirname(__file__), 
                                       'weights/vit_base_patch16_224.augreg_in21k_ft_in1k.bin')
        def filter_fn(state_dict, _):
            if num_classes is None: # use fine-tuned in1k fc head
                return state_dict
            else: # use a new linear
                del state_dict['head.weight']
                del state_dict['head.bias']
                return state_dict
            
        load_checkpoint(res, checkpoint_path, strict=False, filter_fn=filter_fn)
    
    res.eval()
    input_sample = torch.rand(2, 3, 224, 224)
    o1 = res(input_sample)
    
    logger.info(f'make attention prunable')
    make_attention_prunable(res)
    # logger.info(f'make softmax prunable')
    # make_softmax_prunable(res)
    
    res.eval()
    o2 = res(input_sample)
    # print(((o1 - o2) ** 2).sum())
    assert ((o1 - o2) ** 2).sum() < 1e-5
    return res


def make_softmax_prunable(model):
    model.eval()
    input_sample = torch.rand(2, 3, 224, 224).to(get_model_device(model))
    o1 = model(input_sample)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Softmax):
            set_module(model, name, SoftmaxIgnoringZero())
            logger.info(f'make softmax {name} prunable')
            
    model.eval()
    o2 = model(input_sample)
    assert ((o1.logits - o2.logits) ** 2).sum() < 1e-5
    return model


if __name__ == '__main__':
    model = vit_l_16()
    model(torch.rand((1, 3, 224, 224)))

    
    # from utils.dl.common.data_loader import ImageNetDataLoader
    # _, test_loader = ImageNetDataLoader('/data/zql/datasets/imagenet2012/train', '/data/zql/datasets/imagenet2012/val', 512, 8)

    # import torch
    # import tqdm
    # import torch.nn.functional as F
    # def get_accuracy(model, dataloader=test_loader, device='cuda'):
    #     acc = 0
    #     sample_num = 0
        
    #     model.eval()
    #     model = model.to(device)
        
    #     with torch.no_grad():
    #         pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True, leave=False)
    #         for batch_index, (x, y) in pbar:
    #             x, y = x.to(device), y.to(device)
    #             output = model(x)
    #             pred = F.softmax(output, dim=1).argmax(dim=1)
    #             correct = torch.eq(pred, y).sum().item()
    #             acc += correct
    #             sample_num += len(y)
                
    #             pbar.set_description(f'cur_batch_total: {len(y)}, cur_batch_correct: {correct}, '
    #                                  f'cur_batch_acc: {(correct / len(y)):.4f}')

    #     acc /= sample_num
    #     return acc

    # model = model.cuda()
    # print(f'vit_l_16 im1k acc: {get_accuracy(model, test_loader, "cuda")}')
    
    
    # softmax = SoftmaxIgnoringZero()
    
    # x = torch.tensor([[[1, 0, 3], [2, 2, 0]]] * 2).float()
    # print(softmax(x))
    
    
    # model = vit_b_16(True)
    # print(get_accuracy(model))
    
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Softmax):
    #         set_module(model, name, SoftmaxIgnoringZero())
    #         print(f'{name}')
    
    # # print(model)
    # print(get_accuracy(model))
    
    # softmax = SoftmaxIgnoringZero()
    # linear = nn.Linear(20, 10)
    
    # net = nn.Sequential(linear, softmax)
    
    # optimizer = torch.optim.SGD(net.parameters(), lr=10, momentum=0.9)

    # x = torch.rand((64, 20))
    # y_g = torch.rand((64, 10))

    # for _ in range(100):
    #     y = net(x)
    #     # print(y)
        
    #     loss = F.mse_loss(y, y_g)
        
    #     optimizer.zero_grad()
    #     loss.backward()
        
    #     # print(linear.weight.grad)
        
    #     optimizer.step()
        
    #     print(loss)
        
    
    softmax = SoftmaxIgnoringZero()
    
    x = torch.tensor([
        [1, 0, 2],
        [4, 0, 9],
        [0, 0, 0],
        [1, 1, 1]
    ]).float()
    print(softmax(x))
    
    
    x = torch.tensor([
        [1, 2],
        [4, 9],
    ]).float()
    print(softmax(x))