from typing import Dict, Any
from torch import nn
from data.datasets.ab_dataset import ABDataset
from abc import ABC, abstractmethod
from utils.common.log import logger
import json
import os

from utils.common.others import backup_key_codes
from .model import BaseModel
from data import Scenario
from schema import Schema
from utils.common.data_record import write_json


class BaseAlg(ABC):
    
    def __init__(self, models: Dict[str, BaseModel], res_save_dir):
        self.models = models
        self.res_save_dir = res_save_dir
        self.get_required_models_schema().validate(models)

        os.makedirs(res_save_dir)
        logger.info(f'[alg] init alg: {self.__class__.__name__}, res saved in {res_save_dir}')
        
    @abstractmethod
    def get_required_models_schema(self) -> Schema:
        raise NotImplementedError
    
    @abstractmethod
    def get_required_hyp_schema(self) -> Schema:
        raise NotImplementedError
    
    @abstractmethod
    def run(self, 
            scenario: Scenario,
            hyps: Dict) -> Dict[str, Any]:
        """
        return metrics
        """
        
        self.get_required_hyp_schema().validate(hyps)
        
        try:
            write_json(os.path.join(self.res_save_dir, 'hyps.json'), hyps, ensure_obj_serializable=True)
        except:
            with open(os.path.join(self.res_save_dir, 'hyps.txt'), 'w') as f:
                f.write(str(hyps))

        write_json(os.path.join(self.res_save_dir, 'scenario.json'), scenario.to_json())
        
        logger.info(f'[alg] alg {self.__class__.__name__} start running')
        
        backup_key_codes(os.path.join(self.res_save_dir, 'backup_codes'))