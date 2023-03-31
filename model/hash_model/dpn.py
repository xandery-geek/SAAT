import random
import torch
from model.hash_model.base_hash import BaseHashModel


class DPN(BaseHashModel):
    """
    paper: 
    - Deep Polarized Network for Supervised Learning of Accurate Binary Hashing Codes
    - https://www.ijcai.org/Proceedings/2020/115
    """
    def __init__(self, dataset, training=True, **kwargs):
        super().__init__(**kwargs)
        self.model_name = '{}_DPN_{}_{}'.format(dataset, self.backbone, self.bit)
        self.model = self._build_graph()

        self.m = 1
        self.is_single_label = dataset not in {"NUS-WIDE", "MS-COCO", "FLICKR-25K"}

        # init hash center
        self.register_buffer('hash_targets', torch.randint(2, (kwargs['num_class'], self.bit)))
        self.register_buffer('random_center', torch.randint(2, (self.bit, )))
        self.hash_targets = self.get_hash_targets(kwargs['num_class'], self.bit).float().cuda()
        self.random_center = (self.random_center * 2 - 1).float().cuda()

        if training:
            self.num_train = kwargs['num_train']
            self.num_class = kwargs['num_class']

            self.U = torch.zeros(self.num_train, self.bit).float().cuda()  # (N, bit)
            self.Y = torch.zeros(self.num_train, self.num_class).float().cuda()  # (N, class number)
    
    def get_hash_targets(self, num_class, bit, p=0.5):
        hash_targets = torch.zeros(num_class, bit)
        for _ in range(20):
            for index in range(num_class):
                ones = torch.ones(bit)
                sa = random.sample(list(range(bit)), int(bit * p))
                ones[sa] = -1
                hash_targets[index] = ones
        return hash_targets

    def update_hash_targets(self):
        self.U = (self.U.abs() > self.m).float() * self.U.sign()
        self.hash_targets = (self.Y.t() @ self.U).sign()

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
        self.U[index, :] = u.data  # (N, bit)
        self.Y[index, :] = y.float()

        t = self.label2center(y)
        loss = (self.m - u * t).clamp(0).mean()
        return loss

    def forward(self, x, alpha=1):
        return self.model(x, alpha)
