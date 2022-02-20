from model.hash_model.backbone import *
from abc import abstractmethod
import torch


class BaseHashModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bit = kwargs['bit']
        self.backbone = kwargs['backbone']
        self.model_name = ''

    def _build_graph(self):
        if self.backbone == 'AlexNet':
            model = AlexNet(self.bit)
        elif 'VGG' in self.backbone:
            model = VGG(self.backbone, self.bit)
        else:
            model = ResNet(self.backbone, self.bit)
        return model

    @abstractmethod
    def loss_function(self, *args):
        pass
