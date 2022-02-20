import torch
from model.hash_model.base_hash import BaseHashModel


class DPSH(BaseHashModel):
    """
    From DPSH
    """
    def __init__(self, dataset, training=True, **kwargs):
        super().__init__(**kwargs)
        self.model_name = '{}_DPSH_{}_{}'.format(dataset, self.backbone, self.bit)
        self.model = self._build_graph()

        if training:
            self.num_train = kwargs['num_train']
            self.num_class = kwargs['num_class']
            self.yita = 50

            self.U = torch.zeros(self.num_train, self.bit).float().cuda()  # (N, bit)
            self.Y = torch.zeros(self.num_train, self.num_class).float().cuda()  # (N, class number)

    @staticmethod
    def log_trick(x):
        lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]).cuda())
        return lt

    def loss_function(self, u, y, index):
        self.U[index, :] = u.data  # (N, bit)
        self.Y[index, :] = y.float()

        batch_size = len(u)
        similarity = (y @ self.Y.t() > 0).float()  # (B, N)

        b = torch.sign(u)  # binary code (B, bit)
        dot_product = u @ self.U.t() / 2  # inner product of real code (B, N)
        sim_loss = (similarity * dot_product - self.log_trick(dot_product)).sum() / (batch_size * self.num_train)
        reg_term = (b - u).pow(2).sum() / (batch_size * self.num_train)
        loss = -sim_loss + self.yita * reg_term
        return loss

    def forward(self, x, alpha=1):
        return self.model(x, alpha)
