import torch
from model.util import get_alpha
from model.adv_generator.base import BaseGenerator


class SINIFGSMGenerator(BaseGenerator):
    def __init__(self, model, eps, step=1.0, iteration=100, decay=1.0, m=5, targeted=False, record_loss=False):
        super().__init__('SINIFGSM', model, eps, targeted, record_loss)

        self.step = step
        self.iteration = iteration
        self.decay = decay
        self.alpha = self.step / 255
        self.m = m
    
    def forward(self, images, target_code):
        momentum = torch.zeros_like(images).detach().cuda()
        adv_images = images.clone().detach()

        for i in range(self.iteration):
            adv_images.requires_grad = True
            nes_images = adv_images + self.decay * self.alpha * momentum

            alpha = get_alpha(i, self.iteration)
            adv_grad = torch.zeros_like(images).detach().cuda()
            for i in torch.arange(self.m):
                nes_images = nes_images / torch.pow(2, i)
                adv_code = self.model(nes_images, alpha)
                cost = - self.loss(adv_code, target_code)

                adv_grad += torch.autograd.grad(cost, nes_images,
                                                retain_graph=False, 
                                                create_graph=False)[0]
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay * momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
