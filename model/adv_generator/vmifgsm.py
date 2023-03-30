import torch
from model.util import get_alpha
from model.adv_generator.base import BaseGenerator


class VMIFGSMGenerator(BaseGenerator):
    def __init__(self, model, eps, step=1.0, iteration=100, decay=1.0, N=5, beta=3/2, targeted=False, record_loss=False):
        super().__init__('VMIFGSM', model, eps, targeted, record_loss)

        self.step = step
        self.iteration = iteration
        self.decay = decay
        self.N = N
        self.beta = beta
        self.alpha = self.step / 255

    def forward(self, images, target_code):
        momentum = torch.zeros_like(images).detach().cuda()
        v = torch.zeros_like(images).detach().cuda()
        adv_images = images.clone().detach()

        for i in range(self.iteration):
            adv_images.requires_grad = True
            alpha = get_alpha(i, self.iteration)
            adv_code = self.model(adv_images, alpha)
            cost = - self.loss(adv_code, target_code)

            adv_grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            GV_grad = torch.zeros_like(images).detach().cuda()
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-self.eps * self.beta, self.eps * self.beta)
                neighbor_images.requires_grad = True

                alpha = get_alpha(i, self.iteration)
                adv_code = self.model(neighbor_images, alpha)
                cost = - self.loss(adv_code, target_code)

                GV_grad += torch.autograd.grad(cost, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
