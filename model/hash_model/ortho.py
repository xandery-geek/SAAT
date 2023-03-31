import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hash_model.base_hash import BaseHashModel


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, codebook=None, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_centroid={}'.format(
            self.nfeat, self.nclass, self.learn_cent
        )


class Ortho(BaseHashModel):
    """
    paper:
    - One Loss for All: Deep Hashing with a Single Cosine Similarity based Learning Objective
    """
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.model_name = '{}_Ortho_{}_{}'.format(dataset, self.backbone, self.bit)
        self.model = self._build_graph()

        self.is_single_label = dataset not in {"NUS-WIDE", "MS-COCO", "FLICKR-25K"}

        # init codebook
        self.register_buffer('codebook', torch.randint(2, (kwargs['num_class'], self.bit)))
        prob = torch.ones(kwargs['num_class'], self.bit) * 0.5
        self.codebook = (torch.bernoulli(prob) * 2. - 1.).sign().float().cuda()

        self.batchnorm = nn.BatchNorm1d(self.bit, momentum=0.1)
        self.fc = CosSim(self.bit, kwargs['num_class'], self.codebook, learn_cent=False)

        self.ce = 1
        self.s = 8
        self.m = 0.2
        self.m_type = 'cos'
        self.multiclass = dataset in {"NUS-WIDE", "MS-COCO", "FLICKR-25K"}

        self.quan = 0
        self.quan_type = 'cs'
        self.multiclass_loss = 'label_smoothing'
    
    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def loss_function(self, u, y, index):
        code_logits = self.batchnorm(u)
        logits = self.fc(code_logits)
        labels = y
        if self.multiclass:
            margin_logits = self.compute_margin_logits(logits, labels)

            if self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce = loss_ce.mean()
            else:
                raise NotImplementedError(f'unknown method: {self.multiclass_loss}')
        else:
            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)

        if self.quan != 0:
            if self.quan_type == 'cs':
                quantization = (1. - F.cosine_similarity(code_logits, code_logits.detach().sign(), dim=1))
            elif self.quan_type == 'l1':
                quantization = torch.abs(code_logits - code_logits.detach().sign())
            else:  # l2
                quantization = torch.pow(code_logits - code_logits.detach().sign(), 2)

            quantization = quantization.mean()
        else:
            quantization = torch.tensor(0.).to(code_logits.device)

        loss = self.ce * loss_ce + self.quan * quantization
        return loss

    def forward(self, x, alpha=1):
        return self.model(x, alpha)
