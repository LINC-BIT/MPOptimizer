import torch
from torch import nn 
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from einops import rearrange
from dnns.yolov3.yolo_fpn import YOLOFPN
from dnns.yolov3.head import YOLOXHead
from utils.dl.common.model import set_module, get_module
from types import MethodType
import os
from utils.common.log import logger


class VisionTransformerYOLOv3(VisionTransformer):
    def forward_head(self, x):
        # print(222)
        return self.head(x)
    
    def forward_features(self, x):
        # print(111)
        return self._intermediate_layers(x, n=[len(self.blocks) // 3 - 1, len(self.blocks) // 3 * 2 - 1, len(self.blocks) - 1])
    
    def forward(self, x, targets=None):
        features = self.forward_features(x)
        return self.head(x, features, targets)
    
    @staticmethod
    def init_from_vit(vit: VisionTransformer):
        res = VisionTransformerYOLOv3()
        
        for attr in dir(vit):
            # if str(attr) not in ['forward_head', 'forward_features'] and not attr.startswith('__'):
            if isinstance(getattr(vit, attr), nn.Module):
                # print(attr)
                try:
                    setattr(res, attr, getattr(vit, attr))
                except Exception as e:
                    print(attr, str(e))
        
        return res
    
    
class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ViTYOLOv3Head(nn.Module):
    def __init__(self, im_size, patch_size, patch_dim, num_classes, use_bigger_fpns, cls_vit_ckpt_path, init_head):
        super(ViTYOLOv3Head, self).__init__()

        self.im_size = im_size
        self.patch_size = patch_size
        
        # target_patch_dim: [256, 512, 512]
        # self.change_patchs_dim = nn.ModuleList([nn.Linear(patch_dim, target_patch_dim) for target_patch_dim in [256, 512, 512]])
        
        # # input: (1, target_patch_dim, 14, 14)
        # # target feature size: {40, 20, 10}
        # self.change_features_size = nn.ModuleList([
        #     self.get_change_feature_size(cin, cout, t) for t, cin, cout in zip([40, 20, 10], [256, 512, 512], [256, 512, 512])
        # ])
        
        embed_dim = 768
        self.before_fpns = nn.ModuleList([
            # nn.Sequential(
            #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            #     nn.GroupNorm(embed_dim),
            #     nn.GELU(),
            #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            # ),
            
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            ),
            
            nn.Identity(),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        if use_bigger_fpns == 1:
            logger.info('use 421x fpns')
            self.before_fpns = nn.ModuleList([
                
                nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    Norm2d(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                ),
                
                nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                ),
                
                nn.Identity(),
                
                # nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        if use_bigger_fpns == -1:
            logger.info('use 1/0.5/0.25x fpns')
            self.before_fpns = nn.ModuleList([
                
                # nn.Sequential(
                #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                # ),
                
                nn.Identity(),
                
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=2, stride=2))
            ])
        
        # self.fpn = YOLOFPN()
        self.fpn = nn.Identity()
        self.head = YOLOXHead(num_classes, in_channels=[768, 768, 768], act='lrelu')
        
        if init_head:
            logger.info('init head')
            self.load_pretrained_weight(cls_vit_ckpt_path)
        else:
            logger.info('do not init head')
        
    def load_pretrained_weight(self, cls_vit_ckpt_path):
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'yolox_darknet.pth'))
        # for k in [f'head.cls_preds.{i}.{j}' for i in [0, 1, 2] for j in ['weight', 'bias']]:
        #     del ckpt['model'][k]
        removed_k = [f'head.cls_preds.{i}.{j}' for i in [0, 1, 2] for j in ['weight', 'bias']]
        for k, v in ckpt['model'].items():
            if 'backbone.backbone' in k:
                removed_k += [k]
            if 'head.stems' in k and 'conv.weight' in k:
                removed_k += [k]
        for k in removed_k:
            del ckpt['model'][k]
        # print(ckpt['model'].keys())
        
        new_state_dict = {}
        for k, v in ckpt['model'].items():
            new_k = k.replace('backbone', 'fpn')
            new_state_dict[new_k] = v
            
        # cls_vit_ckpt = torch.load(cls_vit_ckpt_path)
        # for k, v in cls_vit_ckpt['main'].named_parameters():
        #     if not 'qkv.abs' not in k:
        #         continue
            
        #     new_state_dict[k] = v
        #     logger.info(f'load {k} from cls vit ckpt')
            
        self.load_state_dict(new_state_dict, strict=False)
        
    def get_change_feature_size(self, in_channels, out_channels, target_size):
        H, W = self.im_size
        GS = H // self.patch_size # 14
        
        if target_size == GS:
            return nn.Identity()
        elif target_size < GS:
            return nn.AdaptiveMaxPool2d((target_size, target_size))
        else:
            return {
                20: nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=0),
                    # nn.BatchNorm2d(out_channels),
                    # nn.ReLU(),
                    nn.AdaptiveMaxPool2d((target_size, target_size))
                ),
                40: nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=3, padding=1),
                    # nn.BatchNorm2d(out_channels),
                    # nn.ReLU(),
                )
            }[target_size]

    def forward(self, input_images, x, targets=None):
        # print(111)
        # NOTE: YOLOX backbone (w/o FPN) output, or FPN input: {'dark3': torch.Size([4, 256, 40, 40]), 'dark4': torch.Size([4, 512, 20, 20]), 'dark5': torch.Size([4, 512, 10, 10])}
        # NOTE: YOLOXHead input: [torch.Size([4, 128, 40, 40]), torch.Size([4, 256, 20, 20]), torch.Size([4, 512, 10, 10])]
        # print(x)
        
        # print([i.size() for i in x])
        x = [i[:, 1:] for i in x]
        x = [i.permute(0, 2, 1).reshape(input_images.size(0), -1, 14, 14) for i in x] # 14 is hardcode, obtained from timm.layers.patch_embed.py
        # print([i.size() for i in x])
        # exit()
        
        # NOTE: old
        # x[0]: torch.Size([1, 196, 768])
        # H, W = self.im_size
        # GS = H // self.patch_size # 14
        # xs = [cpd(x) for x, cpd in zip(xs, self.change_patchs_dim)] # (1, 196, target_patch_dim)
        # xs = [rearrange(x, "b (h w) c -> b c h w", h=GS) for x in xs] # (1, target_patch_dim, 14, 14)
        
        # xs = [cfs(x) for x, cfs in zip(xs, self.change_features_size)]
        # print([i.size() for i in xs])
        # ----------------
        
        xs = [before_fpn(x[-1]) for i, before_fpn in zip(x, self.before_fpns)]
        
        # print([i.size() for i in xs])
        # exit()
        # [torch.Size([1, 768, 28, 28]), torch.Size([1, 768, 14, 14]), torch.Size([1, 768, 7, 7])]
        
        
        xs = self.fpn(xs)
        # print('before head', [i.size() for i in xs])
        xs = tuple(xs)
        
        if targets is not None:
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(xs, targets, input_images)
            return {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            
        return self.head(xs)
    
    
class ViTYOLOv3Head2(nn.Module):
    def __init__(self, im_size, patch_size, patch_dim, num_classes, use_bigger_fpns):
        super(ViTYOLOv3Head2, self).__init__()

        self.im_size = im_size
        self.patch_size = patch_size
        
        # target_patch_dim: [256, 512, 512]
        # self.change_patchs_dim = nn.ModuleList([nn.Linear(patch_dim, target_patch_dim) for target_patch_dim in [256, 512, 512]])
        
        # # input: (1, target_patch_dim, 14, 14)
        # # target feature size: {40, 20, 10}
        # self.change_features_size = nn.ModuleList([
        #     self.get_change_feature_size(cin, cout, t) for t, cin, cout in zip([40, 20, 10], [256, 512, 512], [256, 512, 512])
        # ])
        
        embed_dim = 768
        self.before_fpns = nn.ModuleList([
            # nn.Sequential(
            #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            #     nn.GroupNorm(embed_dim),
            #     nn.GELU(),
            #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            # ),
            
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            ),
            
            nn.Identity(),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        if use_bigger_fpns:
            logger.info('use 8/4/2x fpns')
            self.before_fpns = nn.ModuleList([
                
                nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    Norm2d(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                ),
                
                nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                ),
                
                nn.Identity(),
                
                # nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        
        # self.fpn = YOLOFPN()
        self.fpn = nn.Identity()
        self.head = YOLOXHead(num_classes, in_channels=[768, 768, 768], act='lrelu')
        
        self.load_pretrained_weight()
        
    def load_pretrained_weight(self):
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'yolox_darknet.pth'))
        # for k in [f'head.cls_preds.{i}.{j}' for i in [0, 1, 2] for j in ['weight', 'bias']]:
        #     del ckpt['model'][k]
        removed_k = [f'head.cls_preds.{i}.{j}' for i in [0, 1, 2] for j in ['weight', 'bias']]
        for k, v in ckpt['model'].items():
            if 'backbone.backbone' in k:
                removed_k += [k]
            if 'head.stems' in k and 'conv.weight' in k:
                removed_k += [k]
        for k in removed_k:
            del ckpt['model'][k]
        # print(ckpt['model'].keys())
        
        new_state_dict = {}
        for k, v in ckpt['model'].items():
            new_k = k.replace('backbone', 'fpn')
            new_state_dict[new_k] = v
        self.load_state_dict(new_state_dict, strict=False)
        
    def get_change_feature_size(self, in_channels, out_channels, target_size):
        H, W = self.im_size
        GS = H // self.patch_size # 14
        
        if target_size == GS:
            return nn.Identity()
        elif target_size < GS:
            return nn.AdaptiveMaxPool2d((target_size, target_size))
        else:
            return {
                20: nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=0),
                    # nn.BatchNorm2d(out_channels),
                    # nn.ReLU(),
                    nn.AdaptiveMaxPool2d((target_size, target_size))
                ),
                40: nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=3, padding=1),
                    # nn.BatchNorm2d(out_channels),
                    # nn.ReLU(),
                )
            }[target_size]

    def forward(self, input_images, x, targets=None):
        # print(111)
        # NOTE: YOLOX backbone (w/o FPN) output, or FPN input: {'dark3': torch.Size([4, 256, 40, 40]), 'dark4': torch.Size([4, 512, 20, 20]), 'dark5': torch.Size([4, 512, 10, 10])}
        # NOTE: YOLOXHead input: [torch.Size([4, 128, 40, 40]), torch.Size([4, 256, 20, 20]), torch.Size([4, 512, 10, 10])]
        # print(x)
        
        # print([i.size() for i in x])
        x = [i[:, 1:] for i in x]
        x = [i.permute(0, 2, 1).reshape(input_images.size(0), -1, 14, 14) for i in x] # 14 is hardcode, obtained from timm.layers.patch_embed.py
        # print([i.size() for i in x])
        # exit()
        
        # NOTE: old
        # x[0]: torch.Size([1, 196, 768])
        # H, W = self.im_size
        # GS = H // self.patch_size # 14
        # xs = [cpd(x) for x, cpd in zip(xs, self.change_patchs_dim)] # (1, 196, target_patch_dim)
        # xs = [rearrange(x, "b (h w) c -> b c h w", h=GS) for x in xs] # (1, target_patch_dim, 14, 14)
        
        # xs = [cfs(x) for x, cfs in zip(xs, self.change_features_size)]
        # print([i.size() for i in xs])
        # ----------------
        
        xs = [before_fpn(i) for i, before_fpn in zip(x, self.before_fpns)]
        
        # print([i.size() for i in xs])
        # exit()
        # [torch.Size([1, 768, 28, 28]), torch.Size([1, 768, 14, 14]), torch.Size([1, 768, 7, 7])]
        
        
        xs = self.fpn(xs)
        # print('before head', [i.size() for i in xs])
        xs = tuple(xs)
        
        if targets is not None:
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(xs, targets, input_images)
            return {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            
        return self.head(xs)
    
    
def _forward_head(self, x):
    return self.head(x)


# def ensure_forward_head_obj_repoint(self):
#     self.forward_head = MethodType(_forward_head, self)


@torch.no_grad()
def make_vit_yolov3(vit: VisionTransformer, samples: torch.Tensor, patch_size, patch_dim, num_classes, 
                    use_bigger_fpns=False, use_multi_layer_feature=False, cls_vit_ckpt_path=None, init_head=False):
    
    assert cls_vit_ckpt_path is None
    
    # vit -> fpn -> head
    
    # modify vit.forward() to make it output middle features
    # vit.forward_features = partial(vit._intermediate_layers, 
    #                       n=[len(vit.blocks) // 3 - 1, len(vit.blocks) // 3 * 2 - 1, len(vit.blocks) - 1])
    # vit.forward_head = _forward_head
    # vit.__deepcopy__ = MethodType(ensure_forward_head_obj_repoint, vit)
    
    vit = VisionTransformerYOLOv3.init_from_vit(vit)
    
    if not use_multi_layer_feature:
        set_module(vit, 'head', ViTYOLOv3Head(
                im_size=(samples.size(2), samples.size(3)),
                patch_size=patch_size,
                patch_dim=patch_dim,
                num_classes=num_classes,
                use_bigger_fpns=use_bigger_fpns,
                cls_vit_ckpt_path=cls_vit_ckpt_path,
                init_head=init_head
            ))

            
    else:
        raise NotImplementedError
        logger.info('use multi layer feature')
        set_module(vit, 'head', ViTYOLOv3Head2(
                im_size=(samples.size(2), samples.size(3)),
                patch_size=patch_size,
                patch_dim=patch_dim,
                num_classes=num_classes,
                use_bigger_fpns=use_bigger_fpns,
                cls_vit_ckpt_path=cls_vit_ckpt_path
            ))
    
    # print(vit)
    
    vit.eval()
    output = vit(samples)
    # print([oo.size() for oo in output])
    assert len(output) == samples.size(0) and output[0].size(1) == num_classes + 5, f'{[oo.size() for oo in output]}, {num_classes}'
    
    return vit
    

if __name__ == '__main__':
    from dnns.vit import vit_b_16
    vit_b_16 = vit_b_16()
    make_vit_yolov3(vit_b_16, torch.rand((1, 3, 224, 224)), 16, 768, 20)
    exit()
    
    from types import MethodType
    
    class Student(object):
        pass
    
    def set_name(self, name):
        self.name = name
        
    def get_name(self):
        print(self.name)

    s1 = Student()
    #将方法绑定到s1和s2实例中
    s1.set_name = MethodType(set_name, s1)
    s1.get_name = MethodType(get_name, s1)
    s1.set_name('s1')
    
    from copy import deepcopy
    s2 = deepcopy(s1)
    s2.get_name()
    
    s2.set_name('s2')
    s1.get_name()
    s2.get_name()