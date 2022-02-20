import math
from model.hash_model.backbone import *
from model.hash_model.base_hash import BaseHashModel


class HashNet(BaseHashModel):
    def __init__(self, dataset, training=True, **kwargs):
        """
        :param dataset: dataset name
        :param num_train: number of train dataloader
        :param num_class:  number of class in dataset
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.model_name = '{}_HashNet_{}_{}'.format(dataset, self.backbone, self.bit)
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

        return loss

    def forward(self, x, alpha=None):
        if self.training:
            self.iter_num += 1
            if self.iter_num % self.step_size == 0:
                self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        else:
            self.scale = alpha if alpha is not None else self.init_scale
        return self.model(x, self.scale)
