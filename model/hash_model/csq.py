import torch
import numpy as np
from scipy.linalg import hadamard
from model.hash_model.base_hash import BaseHashModel


class CSQ(BaseHashModel):
    """
    From CSQ
    Ref:
    - https://github.com/yuanli2333/Hadamard-Matrix-for-hashing
    - https://github.com/swuxyj/DeepHash-pytorch
    """
    def __init__(self, dataset, training=True, **kwargs):
        super().__init__(**kwargs)
        self.model_name = '{}_CSQ_{}_{}'.format(dataset, self.backbone, self.bit)
        self.model = self._build_graph()

        self.is_single_label = dataset not in {"NUS-WIDE", "MS-COCO", "FLICKR-25K"}
        self.hash_targets = self.get_hash_targets(kwargs['num_class'], self.bit).cuda()
        self.random_center = torch.randint(2, (self.bit, )).float().cuda()
        self.random_center = self.random_center * 2 - 1
        self.criterion = torch.nn.BCELoss().cuda()

        if training:
            self.p_lambda = 0.05

    @staticmethod
    def get_hash_targets(n_class, bit):
        ha_d = hadamard(bit)
        ha_2d = np.concatenate((ha_d, -ha_d), 0)

        assert ha_2d.shape[0] >= n_class
        
        hash_targets = torch.from_numpy(ha_2d[:n_class]).float()
        return hash_targets

    def label2center(self, y):
        """
        y: lable, [B, num_class]
        """
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    def loss_function(self, u, y, index):
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        quan_loss = (u.abs() - 1).pow(2).mean()
        loss = center_loss + self.p_lambda * quan_loss
        return loss

    def forward(self, x, alpha=1):
        return self.model(x, alpha)
