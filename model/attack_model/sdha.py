# SDHA: A Smart Adversarial Attack on Deep Hashing Based Image Retrieval

import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.hamming_matching import cal_map, cal_perceptibility
from model.util import get_alpha, load_model, get_attack_model_name, get_database_code
from utils.data_provider import get_data_loader, get_data_label
from utils.util import Logger


def load_optimizer(param):
    return torch.optim.Adam(param, lr=0.01, betas=(0.9, 0.999))


def cal_hamming_dis(b1, b2):
    """
    :param b1: k
    :param b2: n * k
    :return: Hamming distance between b1 and b2
    """
    k = b2.size(1)
    return 0.5 * (k - b1 @ b2.transpose(0, 1))


def surrogate_func(h_hat, h, idx, sigma=5, z=0.5):
    batch_size = h.size(0)
    similarity = g_similarity[idx]  # b * n

    loss = 0
    for i in range(batch_size):
        q = h_hat[i]  # 1 * k
        r_idx = torch.where(similarity[i] > 0)[0]
        r = h[i].unsqueeze(0) if r_idx.size(dim=0) == 0 else g_code[r_idx]  # n_{u} * k
        r = r.sign()
        dis = 0.5 * torch.sum(torch.abs(q - r) * torch.sigmoid(sigma * q * r), dim=1) + \
              torch.sum(1 - torch.sigmoid(sigma * q * r), dim=1)
        w = torch.pow(cal_hamming_dis(q.sign(), r) + 1e-8, -z)
        loss += torch.mean(dis * w)
    return - loss / batch_size


def surrogate_func_targeted(h_hat, h, idx, sigma=10, z=-0.3):
    batch_size = h.size(0)
    similarity = g_similarity[idx]  # b * n

    loss = 0
    for i in range(batch_size):
        q = h_hat[i]  # 1 * k
        r_idx = torch.where(similarity[i] > 0)[0]
        r = g_code[r_idx]  # n_{u} * k
        r = r.sign()
        dis = 0.5 * torch.sum(torch.abs(q - r) * torch.sigmoid(-sigma * q * r), dim=1)
        w = torch.pow(cal_hamming_dis(q.sign(), r) + 1e-8, -z)
        loss += torch.mean(dis * w)
    return loss / batch_size


def sdha_loss(x_hat, x, h_hat, h, idx, alpha=25., targeted=False):
    k = h.size(1)
    alpha = alpha / k
    mse_loss = F.mse_loss(x_hat, x)
    if not targeted:
        surrogate_loss = surrogate_func(h_hat, h, idx)
    else:
        surrogate_loss = surrogate_func_targeted(h_hat, h, idx)
    return mse_loss + alpha * surrogate_loss


def adv_generator(model, x, idx, targeted=False, epochs=100, epsilon=8 / 255., record_loss=False):
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
        loss = sdha_loss(x_hat, x, h_hat, h, idx, targeted=targeted)
        loss.backward()
        optimizer.step()

        x_hat.data = torch.max(torch.min(x_hat.data, x + epsilon), x - epsilon)
        x_hat.data = torch.clamp(x_hat.data, min=0, max=1)  # subject to [0, 1]

        if loss_list is not None and (epoch + 1) % (epochs // 10) == 0:
            loss_list.append(round(loss.item(), 4))
    if loss_list is not None:
        print("loss: {}".format(loss_list))
    return h.cpu().sign(), h_hat.detach().cpu().sign(), x_hat.detach().cpu()


def theory_attack(model, x, idx):
    x = x.cuda()
    h = model(x)
    h = h.detach()
    batch_size = h.size(0)
    similarity = g_similarity[idx]  # b * n

    h_hat = torch.zeros(h.size())
    for i in range(batch_size):
        r_idx = torch.where(similarity[i] > 0)[0]
        r = h[i].unsqueeze(0) if r_idx.size(dim=0) == 0 else g_code[r_idx]  # n_{u} * k
        r = r.sign()
        h_hat[i] = -torch.mean(r, dim=0).sign()
    return h.cpu().sign(), h_hat.cpu()


def theory_attack_targeted(model, x, idx):
    x = x.cuda()
    h = model(x)
    h = h.detach()
    batch_size = h.size(0)
    similarity = g_similarity[idx]  # b * n

    h_hat = torch.zeros(h.size())
    for i in range(batch_size):
        r_idx = torch.where(similarity[i] > 0)[0]
        r = g_code[r_idx]  # n_{u} * k
        r = r.sign()
        h_hat[i] = torch.mean(r, dim=0).sign()
    return h.cpu().sign(), h_hat.cpu()


g_code, g_similarity = torch.tensor(0), torch.tensor(0)


def sdha(args, targeted=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    method = 'SDHA'
    # load model
    attack_model = get_attack_model_name(args)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, _ = get_data_loader(args.data_dir, args.dataset, 'database',
                                         args.batch_size, shuffle=False)
    train_loader, num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                              args.batch_size, shuffle=False)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                     args.batch_size, shuffle=False)

    train_label = get_data_label(args.data_dir, args.dataset, 'train')
    test_label = get_data_label(args.data_dir, args.dataset, 'test')

    # generate global code
    global g_code, g_similarity
    g_code = torch.zeros(num_train, args.bit).float().cuda()

    for x, _, idx in train_loader:
        g_code[idx] = model(x.cuda()).detach()

    # generate global similarity matrix
    train_label_tensor = torch.from_numpy(train_label).float().cuda()
    if targeted:
        # load target label
        target_label_path = 'log/target_label_{}.txt'.format(args.dataset)
        if os.path.exists(target_label_path):
            target_label = np.loadtxt(target_label_path, dtype=np.int)
        else:
            raise ValueError('Please generate target_label before attack!')
        target_label_tensor = torch.from_numpy(target_label).float().cuda()
        g_similarity = target_label_tensor @ train_label_tensor.transpose(0, 1)  # N_test * N_train
    else:
        test_label_tensor = torch.from_numpy(test_label).float().cuda()
        g_similarity = test_label_tensor @ train_label_tensor.transpose(0, 1)  # N_test * N_train
        target_label = test_label

    # attack
    perceptibility = torch.tensor([0, 0, 0], dtype=torch.float)
    query_code_arr, adv_code_arr = None, None
    for _, (x, _, idx) in enumerate(tqdm(test_loader)):
        h, h_hat, x_hat = adv_generator(model, x, idx, targeted=targeted, epochs=args.iteration)
        # h, h_hat = theory_attack_targeted(model, x, idx)
        # h, h_hat = theory_attack(model, x, idx)
        query_code_arr = h.numpy() if query_code_arr is None else np.concatenate((query_code_arr, h.numpy()))
        adv_code_arr = h_hat.numpy() if adv_code_arr is None else np.concatenate((adv_code_arr, h_hat.numpy()))
        perceptibility += cal_perceptibility(x, x_hat) * x.size(0)

    database_code, database_label = get_database_code(model, database_loader, attack_model)

    # save code
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), adv_code_arr)

    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    logger.log('perceptibility: {}'.format(perceptibility / num_test))

    # calculate map
    ori_map = cal_map(database_code, query_code_arr, database_label, test_label, 5000)
    logger.log('Ori MAP(retrieval database): {:.5f}'.format(ori_map))

    if targeted:
        adv_map = cal_map(database_code, adv_code_arr, database_label, target_label, 5000)
        logger.log('SDHA t-MAP(retrieval database): {:.5f}'.format(adv_map))
    else:
        adv_map = cal_map(database_code, adv_code_arr, database_label, test_label, 5000)
        logger.log('SDHA MAP(retrieval database): {:.5f}'.format(adv_map))
