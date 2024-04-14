import torch
from torch import nn 


def to(x, device):
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)
    else:
        x = x.to(device)
    return x


def get_cur_acc(testset, hyps, model, shuffle, iter_index):
    from data import split_dataset, build_dataloader
    cur_test_batch_dataset = split_dataset(testset, hyps['val_batch_size'], iter_index)[0]
    cur_test_batch_dataloader = build_dataloader(cur_test_batch_dataset, hyps['train_batch_size'], hyps['num_workers'], False, shuffle)
    cur_acc = model.get_accuracy(cur_test_batch_dataloader)
    return cur_acc
    