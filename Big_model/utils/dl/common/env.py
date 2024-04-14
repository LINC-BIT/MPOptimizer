import random
import torch
import numpy as np
import os
from ...common.log import logger


def set_random_seed(seed: int):
    """Fix all random seeds in common Python packages (`random`, `torch`, `numpy`). 
    Recommend to use before all codes to ensure reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def create_tbwriter(log_dir, launch_tbboard=False):
    if launch_tbboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_log = SummaryWriter(log_dir)
        from tensorboard import program
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        url = tb.launch()
        logger.info(f'launch tensorboard in {url}')
        
    logger.info(f'tensorboard --logdir="{log_dir}"')
    
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir)
    
    return tb_writer
    