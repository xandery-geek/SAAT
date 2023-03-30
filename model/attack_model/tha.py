# THA: Targeted Attack and Defense for Deep Hashing

import os
import torch
import numpy as np
from torch import nn
from utils.data_provider import get_data_loader, get_data_label, get_classes_num
from utils.hamming_matching import cal_map
from utils.util import Logger
from model.util import get_alpha, get_attack_model_name, load_model, get_database_code, generate_code
from tqdm import tqdm


class PrototypeNet(nn.Module):
    def __init__(self, bit, num_classes):
        super(PrototypeNet, self).__init__()

        self.feature = nn.Sequential(nn.Linear(num_classes, 4096),
                                     nn.ReLU(True), nn.Linear(4096, 512))
        self.hashing = nn.Sequential(nn.Linear(512, bit), nn.Tanh())

    def forward(self, label):
        f = self.feature(label)
        h = self.hashing(f)
        return h


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp, sn):
        ap = torch.clamp_min(- sp.detach() + 2, min=0.)
        an = torch.clamp_min(sn.detach() + 2, min=0.)

        logit_p = - ap * sp * self.gamma
        logit_n = an * sn * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss


def similarity_pn(batch_feature, features, batch_label, labels, bit):
    similarity_matrix = batch_feature @ features.transpose(1, 0)
    similarity_matrix = similarity_matrix / bit
    label_matrix = (batch_label.mm(labels.t()) > 0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


def target_adv_loss(adv_code, target_code):
    loss = -torch.mean(adv_code * target_code)
    return loss


def adv_generator(model, query, target_code, epsilon, step=1, iteration=100, randomize=False):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        alpha = get_alpha(i, iteration)
        adv_code = model(query + delta, alpha)
        loss = target_adv_loss(adv_code, target_code.detach())
        loss.backward()

        # delta.data = delta - step * delta.grad.detach()
        # delta.data = delta - step * delta.grad.detach() / (torch.norm(delta.grad.detach(), 2) + 1e-9)
        delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

    return query + delta.detach()


def tha(args, epsilon=8 / 255.):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    method = 'THA'

    # load model
    attack_model = get_attack_model_name(args)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    num_classes = get_classes_num(args.dataset)
    database_loader, num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                    args.batch_size, shuffle=False)
    train_loader, num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                              args.batch_size, shuffle=True)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    test_label = get_data_label(args.data_dir, args.dataset, 'test')
    database_label = get_data_label(args.data_dir, args.dataset, 'database')
    database_code, _ = get_database_code(model, database_loader, attack_model)

    # get unique label
    unique_label = np.unique(database_label, axis=0)

    if not args.adv:
        pnet_path = 'checkpoint/PrototypeNet_{}.pth'.format(attack_model)
    else:
        pnet_path = 'checkpoint/{}_PrototypeNet_{}.pth'.format(args.adv_method, attack_model)

    if os.path.exists(pnet_path):
        pnet = load_model(pnet_path)
    else:
        print("Training PrototypeNet")
        pnet = PrototypeNet(args.bit, num_classes).cuda()
        pnet_optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-4, betas=(0.5, 0.999))
        epochs, steps = 100, 300
        lr_steps = epochs * steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(pnet_optimizer,
                                                         milestones=(lr_steps / 2, lr_steps * 3 / 4), gamma=0.1)
        circle_loss = CircleLoss(m=0, gamma=1)

        # hash codes of training set
        train_code, train_label = generate_code(model, train_loader)
        train_code, train_label = torch.from_numpy(train_code).cuda(), torch.from_numpy(train_label).cuda()

        for epoch in range(epochs):
            for i in range(steps):
                select_index = np.random.choice(range(unique_label.shape[0]), size=args.batch_size)
                batch_target_label = unique_label[select_index]
                batch_target_label = torch.from_numpy(batch_target_label).float().cuda()
                pnet_optimizer.zero_grad()
                target_hash_label = pnet(batch_target_label)
                sp, sn = similarity_pn(target_hash_label, train_code, batch_target_label, train_label, args.bit)
                logloss = circle_loss(sp, sn) / args.batch_size
                regterm = (torch.sign(target_hash_label) - target_hash_label).pow(2).sum() / (1e4 * args.batch_size)
                loss = logloss + regterm
                loss.backward()
                pnet_optimizer.step()

                if (i+1) % 300 == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, logloss:{:.5f}, regterm: {:.5f}'.
                          format(epoch, i, scheduler.get_last_lr()[0], logloss, regterm))
                scheduler.step()

        torch.save(pnet, pnet_path)
        pnet.eval()

    target_label_path = 'log/target_label_{}.txt'.format(args.dataset)
    if os.path.exists(target_label_path):
        target_label = np.loadtxt(target_label_path, dtype=np.int)
    else:
        raise ValueError('Please generate target_label before attack!')

    adv_code_arr = np.zeros([num_test, args.bit], dtype=np.float32)
    query_code_arr = np.zeros([num_test, args.bit], dtype=np.float32)
    prototype_code_arr = np.zeros((num_test, args.bit), dtype=np.float)
    for it, (query, _, idx) in enumerate(tqdm(test_loader)):
        query = query.cuda()

        batch_target_label = target_label[idx.numpy(), :]
        batch_target_label = torch.from_numpy(batch_target_label).float().cuda()

        batch_prototype_code = pnet(batch_target_label)
        prototype_code = torch.sign(batch_prototype_code)
        prototype_code_arr[idx.numpy(), :] = prototype_code.cpu().data.numpy()
        adv_query = adv_generator(model, query, prototype_code, epsilon, iteration=args.iteration)

        adv_code_arr[idx.numpy(), :] = model(adv_query).sign().cpu().data.numpy()
        query_code_arr[idx.numpy(), :] = model(query).sign().cpu().data.numpy()

    # save code
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), adv_code_arr)

    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    # t-MAP
    t_map = cal_map(database_code, adv_code_arr, database_label, target_label, 5000)
    logger.log('THA t-MAP(retrieval database): {:.5f}'.format(t_map))
    t_map = cal_map(database_code, prototype_code_arr, database_label, target_label, 5000)
    logger.log('Theory t-MAP(retrieval database): {:.5f}'.format(t_map))
