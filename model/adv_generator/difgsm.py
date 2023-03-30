import torch
from model.util import get_alpha
from model.adv_generator.base import BaseGenerator
import torch.nn.functional as F

class DIFGSMGenerator(BaseGenerator):
    def __init__(self, model, eps, step=1.0, iteration=100, decay=0.0, resize_rate=0.9, 
                 diversity_prob=0.5, random_start=True, targeted=False, record_loss=False):
        super().__init__('DIFGSM', model, eps, targeted, record_loss)

        self.step = step
        self.iteration = iteration
        self.decay = decay
        self.alpha = self.step / 255
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x    

    def forward(self, images, target_code):
        momentum = torch.zeros_like(images).detach().cuda()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.iteration):
            adv_images.requires_grad = True

            alpha = get_alpha(i, self.iteration)
            adv_code = self.model(self.input_diversity(adv_images), alpha)
            cost = - self.loss(adv_code, target_code)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
