from abc import ABCMeta, abstractmethod
import copy
from kt_unlearn.core.base import Configurable
from kt_unlearn.data.datasets.Dataset import DatasetWrapper
from kt_unlearn.utils.config.global_ctx import Global
from kt_unlearn.utils.config.local_ctx import Local

class Unlearner(Configurable, metaclass=ABCMeta):

    def __init__(self, global_ctx: Global, local_ctx):
        if hasattr(local_ctx,'dataset'):
            self.dataset = local_ctx.dataset
        if hasattr(local_ctx,'predictor'):
            self.predictor = local_ctx.predictor
            self.device = self.predictor.device
        else:
            self.predictor = 'global'

        super().__init__(global_ctx, local_ctx)   

    def unlearn(self):
        self.__preprocess__()
        self.info('Unlearning copyed predictor: '+str(self.predictor))
        new_model = self.__unlearn__()
        self.__postprocess__()
        return new_model


    def __preprocess__(self):
        pass

    @abstractmethod
    def __unlearn__(self):
        pass

    def __postprocess__(self):
        pass
        
