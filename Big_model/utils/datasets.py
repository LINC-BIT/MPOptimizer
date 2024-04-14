import dg.domainbed.datasets.datasets as D
import torch
import numpy as np
import os


def load_datasets_of_all_domains(dataset_name, data_dir):
    datasets = vars(D)[dataset_name](data_dir)
    return datasets


def load_dataset_of_a_domain(dataset_name, domain_index, data_dir):
    datasets = vars(D)[dataset_name](data_dir)
    return datasets[domain_index]


def load_online_data(dataset_name, data_config, data_dir):
    datasets = load_datasets_of_all_domains(dataset_name, data_dir)

    res_x, res_y = None, None
    dataset_anchors = [0] * len(datasets.ENVIRONMENTS)
    
    for domain_index, n_samples in data_config:
        dataset = datasets[domain_index]
        
        x, y = dataset[dataset_anchors[domain_index]: dataset_anchors[domain_index] + n_samples]
        dataset_anchors[domain_index] += n_samples
        
        if res_x is None:
            res_x = x
            res_y = y
        else:
            res_x = torch.cat([res_x, x])
            res_y = torch.cat([res_y, y])
        
    return res_x, res_y
