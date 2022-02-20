import torch
from model.hash_model.base_hash import BaseHashModel


class DPH(BaseHashModel):
    """
    From HAG
    """
    def __init__(self, dataset, training=True, **kwargs):
        super().__init__(**kwargs)
        self.model_name = '{}_DPH_{}_{}'.format(dataset, self.backbone, self.bit)
        self.model = self._build_graph()

        if training:
            self.num_train = kwargs['num_train']
            self.num_class = kwargs['num_class']

            self.U = torch.zeros(self.num_train, self.bit).float().cuda()  # (N, bit)
            self.Y = torch.zeros(self.num_train, self.num_class).float().cuda()  # (N, class number)

    def loss_function(self, u, y, index):
        self.U[index, :] = u.data  # (N, bit)
        self.Y[index, :] = y.float()

        batch_size = len(u)
        similarity = (y @ self.Y.t() > 0).float()  # (B, N)
        similarity = 2 * similarity - 1  # range to [-1, 1]
        dot_product = u @ self.U.t() / self.bit  # \frac{1}{K} h h^{T}  range: [-1, 1]

        log_loss = (dot_product - similarity) ** 2
        loss = log_loss.sum() / (self.num_train * batch_size)
        return loss

    def forward(self, x, alpha=1):
        return self.model(x, alpha)
