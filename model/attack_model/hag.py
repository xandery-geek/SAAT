# HAG: Adversarial Examples for Hamming Space Search

import os
import torch
import numpy as np
from tqdm import tqdm
from utils.hamming_matching import cal_map, cal_perceptibility
from model.util import get_alpha, get_attack_model_name, load_model, get_database_code
from utils.data_provider import get_data_loader, get_data_label
from utils.util import Logger


def load_optimizer(param):
    return torch.optim.SGD(param, lr=100, momentum=0.9)


def mask_code(adv_code, ori_code, threshold_t):
    """
    calculate mask code
    :param adv_code: hash code of adversarial example
    :param ori_code: hash code of original image
    :param threshold_t:
    :return: mask code
    """
    # if s[i] == 1: ad_code[i] and or_code[i] are invariant, which means they need optimize
    # if s[i] == 0: ad_code[i] and or_code[i] are zero, which means they already have different signs

    mask = torch.sign((1 + threshold_t) - torch.abs(adv_code - torch.sign(ori_code)))  # w \in {-1, 1}
    mask = (mask + 1) / 2  # w \in {0, 1}
    return mask


def hag_loss(adv_code, ori_code, threshold_t=0.5, beta=1.0):
    mask = mask_code(adv_code, ori_code, threshold_t)
    adv_code = mask * adv_code
    ori_code = mask * ori_code

    m = torch.sum(mask, dim=1)
    loss = torch.mean(torch.square(torch.sum(adv_code * ori_code, dim=1) / (m + 1e-8) + 1)) \
           + beta * torch.mean(torch.square(torch.mean(adv_code, dim=1)))
    return loss


def theory_attack(model, x):
    x = x.cuda()
    h = model(x)
    h_hat = -h
    return h.detach().cpu().sign(), h_hat.detach().cpu().sign(), x.detach().cpu()


def adv_generator(model, x, epochs=100, epsilon=8 / 255., record_loss=False):
    x = x.cuda()
    h = model(x)
    x, h = x.detach(), h.detach()

    x_hat = x.clone()
    x_hat.requires_grad_(True)
    optimizer = load_optimizer([x_hat])

    h_hat = None
    loss_list = [] if record_loss else None
    for epoch in range(epochs):
        optimizer.zero_grad()
        alpha = get_alpha(epoch, epochs)
        h_hat = model(x_hat, alpha)
        loss = hag_loss(h_hat, h)
        loss.backward()
        optimizer.step()

        x_hat.data = torch.max(torch.min(x_hat.data, x + epsilon), x - epsilon)
        x_hat.data = torch.clamp(x_hat.data, min=0, max=1)  # subject to [0, 1]

        if loss_list is not None and (epoch + 1) % (epochs // 10) == 0:
            loss_list.append(round(loss.item(), 4))
    if loss_list is not None:
        print("loss: {}".format(loss_list))
    return h.cpu().sign(), h_hat.detach().cpu().sign(), x_hat.detach().cpu()


def hag(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    method = 'HAG'
    # load model
    attack_model = get_attack_model_name(args)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                    args.batch_size, shuffle=False)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    test_labels = get_data_label(args.data_dir, args.dataset, 'test')

    # attack
    perceptibility = torch.tensor([0, 0, 0], dtype=torch.float)
    query_code_arr, adv_code_arr = None, None
    for i, (x, label, idx) in enumerate(tqdm(test_loader)):
        h, h_hat, x_hat = adv_generator(model, x, epochs=args.iteration)
        # h, h_hat, x_hat = theory_attack(model, x)
        query_code_arr = h.numpy() if query_code_arr is None else np.concatenate((query_code_arr, h.numpy()))
        adv_code_arr = h_hat.numpy() if adv_code_arr is None else np.concatenate((adv_code_arr, h_hat.numpy()))
        perceptibility += cal_perceptibility(x, x_hat) * x.size(0)

    database_code, database_label = get_database_code(model, database_loader, attack_model)

    # save code
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), adv_code_arr)

    # calculate map
    ori_map = cal_map(database_code, query_code_arr, database_label, test_labels, 5000)
    adv_map = cal_map(database_code, adv_code_arr, database_label, test_labels, 5000)
    theory_map = cal_map(database_code, -query_code_arr, database_label, test_labels, 5000)

    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    logger.log('perceptibility: {}'.format(perceptibility / num_test))
    logger.log('Ori MAP(retrieval database): {:.5f}'.format(ori_map))
    logger.log('HAG MAP(retrieval database): {:.5f}'.format(adv_map))
    logger.log('Theory MAP(retrieval database): {:.5f}'.format(theory_map))
