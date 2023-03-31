import math
import torch
from model.hash_model.backbone import *
from model.hash_model.base_hash import BaseHashModel


def wasserstein1d(x, y, aggregate=True):
    """Compute wasserstein loss in 1D"""
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    n = x.size(0)
    
    if aggregate:
        z = (x1-y1).view(-1)
        return torch.dot(z, z)/n
    else:
        return (x1-y1).square().sum(0)/n


def quantization_swdc_loss(b, aggregate=True):
    real_b = torch.randn(b.shape).sign().cuda()
    _, dim = b.size()

    if aggregate:
        gloss = wasserstein1d(real_b, b) / dim
    else:
        gloss = wasserstein1d(real_b, b, aggregate=False)

    return gloss


class HSWD(BaseHashModel):
    """
    HashNet with Sliced-Wasserstein-based distributional distance
    paper:
    - One Loss for Quantization: Deep Hashing with DiscreteWasserstein Distributional Matching

    """
    def __init__(self, dataset, training=True, **kwargs):
        """
        :param dataset: dataset name
        :param num_train: number of train dataloader
        :param num_class:  number of class in dataset
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.model_name = '{}_HSWD_{}_{}'.format(dataset, self.backbone, self.bit)
        self.model = self._build_graph()
        if training:
            self.U = torch.zeros(kwargs['num_train'], self.bit).float().cuda()
            self.Y = torch.zeros(kwargs['num_train'], kwargs['num_class']).float().cuda()
            self.alpha = 10.0 / self.bit  # alpha for sigmoid activation

        self.iter_num = 0
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.scale = self.init_scale
        self.quantization_alpha = 0.1

    def loss_function(self, u, y, index):
        """
        :param u: the hash code of inputs
        :param y: labels of inputs
        :param index: index of inputs in total dataset
        :return:
        """
        self.U[index, :] = u.data
        self.Y[index, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = self.alpha * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        quantization_loss = quantization_swdc_loss(u.view(u.size(0), -1))
        loss += self.quantization_alpha * quantization_loss

        return loss

    def forward(self, x, alpha=None):
        if alpha is not None:
            self.scale = alpha
        elif self.training:
            self.iter_num += 1
            if self.iter_num % self.step_size == 0:
                self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        else:
            self.scale = self.init_scale
        return self.model(x, self.scale)
