import torch
import os
from utils.common.log import logger
from utils.common.others import get_cur_time_str
from utils.common.file import ensure_dir


def save_models_dict_for_init(models_dict, exp_entry_file, target_file_name):
    target_file_path = os.path.join(os.path.dirname(exp_entry_file), f'entry_model/{target_file_name}.pt')
    # if os.path.exists(target_file_path):
    #     logger.info(f'model already saved in {target_file_path}, return ({(os.path.getsize(target_file_path) / 1024**2):.3f}MB)')
    #     return target_file_path
    
    ensure_dir(target_file_path)
    torch.save(models_dict, target_file_path)
    logger.info(f'model saved in {target_file_path} ({(os.path.getsize(target_file_path) / 1024**2):.3f}MB)')
    
    return target_file_path


def get_res_save_dir(exp_entry_file, tag=None):
    """
    Design objective: the latest exp result is located in the top of VSCode file explorer (default it is located in the most bottom)
    """
    
    cur_time_str = get_cur_time_str()
    day, time = cur_time_str[0: 8], cur_time_str[8: ]

    base_p = os.path.join(os.path.dirname(exp_entry_file), f'results/{os.path.basename(exp_entry_file)}')
    p = os.path.join(base_p, day)

    if not os.path.exists(p):
        t = 0
    else:
        t = len(os.listdir(p))
    t = f'{(999999 - t):06d}'
    
    if tag is None:
        p = os.path.join(p, f'{t}-{time}')
    else:
        p = os.path.join(p, f'{t}-{time}-{tag}')
        
    return p
