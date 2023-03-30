from abc import ABC, abstractclassmethod


class BaseGenerator(ABC):
    def __init__(self, name, model, eps, targeted=False, record_loss=False):
        self.name = name
        self.model = model
        self.eps = eps
        self.targeted = targeted
        self.record_loss = record_loss

        if targeted:
            self.loss = self.adv_loss_targeted
        else:
            self.loss = self.adv_loss
    
    def __call__(self, *args):
        return self.forward(*args)

    @abstractclassmethod
    def forward(self, images, target_code):
        pass
    
    @staticmethod
    def adv_loss(adv_code, target_code):
        # adversarial loss
        # loss = torch.mean(adv_code * target_code)

        # a better version
        sim = adv_code * target_code
        w = (sim > -0.5).int()
        m = w.sum()
        sim = w * (sim + 2) * sim
        loss = sim.sum()/m
        return loss

    @staticmethod
    def adv_loss_targeted(adv_code, target_code):
        # loss = - torch.mean(adv_code * target_code)
        sim = adv_code * target_code
        w = (sim < 0.5).int()
        m = w.sum()
        sim = w * (sim + 2) * sim
        loss = -sim.sum()/m
        return loss
