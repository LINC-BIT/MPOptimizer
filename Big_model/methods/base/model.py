import torch
from abc import ABC, abstractmethod
from utils.common.file import ensure_dir
from utils.common.log import logger
import time
from typing import List


class BaseModel(ABC):
    def __init__(self,
                 name: str,
                 models_dict_path: str,
                 device: str):
        
        self.name = name
        self.models_dict_path = models_dict_path
        self.models_dict = torch.load(models_dict_path, map_location=device)
        
        self.device = device
        
        assert set(self.get_required_model_components()) <= set(list(self.models_dict.keys()))

        self.to(device)
        
        logger.info(f'[model] init model: {dict(name=name, components=self.get_required_model_components())}')
        logger.debug(self.models_dict)
    
    @abstractmethod
    def get_required_model_components(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_accuracy(self, test_loader, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, x, *args, **kwargs):
        pass
    
    def save_model(self, p: str):
        logger.debug(f'[model] save model: {self.name}')
        ensure_dir(p)
        torch.save(self.models_dict, p)
    
    def load_model(self, p: str):
        logger.debug(f'[model] load model: {self.name}, from {p}')
        self.models_dict = torch.load(p, map_location=self.device)
    
    def to(self, device):
        logger.debug(f'[model] to device: {device}')
        for k, v in self.models_dict.items():
            try:
                self.models_dict[k] = v.to(device)
            except Exception as e:
                pass

    def to_eval_mode(self, verbose=False):
        if verbose:
            logger.info(f'[model] to eval mode')
        for k, v in self.models_dict.items():
            try:
                self.models_dict[k].eval()
            except Exception as e:
                pass

    def to_train_mode(self, verbose=False):
        if verbose:
            logger.info(f'[model] to train mode')
        for k, v in self.models_dict.items():
            try:
                self.models_dict[k].train()
            except Exception as e:
                pass
    