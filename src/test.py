import argparse
import glob
import numpy as np
import os
import pprint
import torch
import torchvision
import tqdm

from glob import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from utils import load_txt, accuracy, create_barplot, get_fname, AverageMeter
from src.models.mobilevit import mobile_vit_xx_small
from src.models.resnet_gn import resnext50_32x4d
from dataset import CIFAR10C
import sys
sys.path.append("./..")
import tent


CORRUPTIONS = load_txt('./src/corruptions.txt')
MEAN = [0.49139968, 0.48215841, 0.44653091]
STD  = [0.24703223, 0.24348513, 0.26158784]


def main(opt, weight_path :str):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # model
    if opt.arch =='resnet':
        model = mobile_vit_xx_small()
    else:
        raise ValueError()
    try:
        # model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        net_dict = model.state_dict()
        print(net_dict.keys())
        predict_model = torch.load(weight_path, map_location='cpu')
        print('start')
        # source_model=torch.load('./src/weights/resnet32.pth', map_location='cpu')
        # model.load_state_dict(source_model)
        state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
        #state_dict1 = {k: v for k, v in predict_model.items()}
        # 寻找网络中公共层，并保留预训练参数
        print(state_dict.keys())
        #print(state_dict1.keys())
        net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        model.load_state_dict(net_dict)  # 加载预训练参数
    except:
        model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    accs = dict()
    with tqdm(total=len(opt.corruptions), ncols=80) as pbar:
        for ci, cname in enumerate(opt.corruptions):
            # load dataset
            if cname == 'natural':
                dataset = datasets.CIFAR10(
                    os.path.join(opt.data_root, 'cifar10'),
                    train=False, transform=transform, download=True,
                )
            else:
                dataset = CIFAR10C(
                    os.path.join(opt.data_root, 'cifar10-c'),
                    cname, transform=transform
                )
            loader = DataLoader(dataset, batch_size=opt.batch_size,
                                shuffle=False, num_workers=4)
            
            acc_meter = AverageMeter()
            with torch.no_grad():
                for itr, (x, y) in enumerate(loader):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, dtype=torch.int64, non_blocking=True)

                    z = model(x)
                    loss = F.cross_entropy(z, y)
                    acc, _ = accuracy(z, y, topk=(1, 5))
                    acc_meter.update(acc.item())

            accs[f'{cname}'] = acc_meter.avg

            pbar.set_postfix_str(f'{cname}: {acc_meter.avg:.2f}')
            pbar.update()
    
    avg = np.mean(list(accs.values()))
    accs['avg'] = avg

    pprint.pprint(accs)
    save_name = get_fname(weight_path)
    create_barplot(
        accs, save_name + f' / avg={avg:.2f}',
        os.path.join(opt.fig_dir, save_name+'.png')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--arch',
        type=str, default='resnet',
        help='model name'
    )
    parser.add_argument(
        '--weight_dir',
        type=str,
        help='path to the dicrectory containing model weights',
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        help='path to the dicrectory containing model weights',
    )
    parser.add_argument(
        '--fig_dir',
        type=str, default='figs',
        help='path to the dicrectory saving output figure',
    )
    parser.add_argument(
        '--data_root',
        type=str, default='/home/tanimu/data',
        help='root path to cifar10-c directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int, default=32,
        help='batch size',
    )
    parser.add_argument(
        '--corruptions',
        type=str, nargs='*',
        default=CORRUPTIONS,
        help='testing corruption types',
    )
    parser.add_argument(
        '--gpu_id',
        type=str, default=0,
        help='gpu id to use'
    )

    opt = parser.parse_args()

    if opt.weight_path is not None:
        main(opt, opt.weight_path)
    elif opt.weight_dir is not None:
        for path in glob(f'./{opt.weight_dir}/*.pth'):
            print('\n', path)
            main(opt, path)
    else:
        raise ValueError("Please specify weight_path or weight_dir option.")
    
#python ./src/test.py --arch model --weight_path ./src/weights/model.pth --data_root ./data_set/CIFAR-10-C --fig_dir ./src/figs