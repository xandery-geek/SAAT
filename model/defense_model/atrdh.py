import os
import torch
import numpy as np
from utils.data_provider import get_data_loader, get_classes_num, get_data_label
from model.util import load_model, generate_code_ordered
from utils.util import check_dir
from model.attack_model.tha import PrototypeNet, CircleLoss, similarity


def cal_similarity(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    S = 2 * S - 1
    return S


def target_adv_loss(noisy_output, target_hash):
    loss = -torch.mean(noisy_output * target_hash)
    # loss = noisy_output * target_hash
    # loss = (loss -2)*loss
    # loss = torch.mean(loss)
    return loss


def target_hash_adv(model, query, target_hash, epsilon, step=2, iteration=7, randomize=True):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        noisy_output = model(query + delta)
        loss = target_adv_loss(noisy_output, target_hash.detach())
        loss.backward()

        delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()
    return query + delta.detach()


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def atrdh(args, epsilon=8 / 255.0, iteration=7):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    num_classes = get_classes_num(args.dataset)
    train_loader, num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                              args.batch_size, shuffle=True)

    database_label = get_data_label(args.data_dir, args.dataset, 'database')
    database_label = torch.from_numpy(database_label).float()
    target_label = database_label.unique(dim=0)

    pnet = PrototypeNet(args.bit, num_classes).cuda()
    circle_loss = CircleLoss(m=0, gamma=1)
    pnet_optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-4, betas=(0.5, 0.999))
    hash_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    lr_steps = args.epochs * len(train_loader)
    pnet_scheduler = torch.optim.lr_scheduler.MultiStepLR(pnet_optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                          gamma=0.1)
    hash_scheduler = torch.optim.lr_scheduler.MultiStepLR(hash_optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                          gamma=0.1)

    train_code, train_label = generate_code_ordered(model, train_loader, num_train, args.bit,
                                                    get_classes_num(args.dataset))

    U_ben = torch.zeros(num_train, args.bit).cuda()

    pnet.train()
    model.train()

    # adversarial training
    for epoch in range(args.epochs):
        for it, data in enumerate(train_loader):
            x, y, index = data
            x, y = x.cuda(), y.cuda()
            batch_size_ = index.size(0)

            ben_code = model(x)
            train_code[index.numpy(), :] = torch.sign(ben_code.detach())

            select_index = np.random.choice(range(target_label.size(0)), size=batch_size_)
            batch_target_label = target_label.index_select(0, torch.from_numpy(select_index)).cuda()

            # optimize pnet
            set_requires_grad(pnet, True)
            pnet_optimizer.zero_grad()
            batch_target_code = pnet(batch_target_label)
            sp, sn = similarity(batch_target_code, train_code, batch_target_label, train_label, args.bit)
            logloss = circle_loss(sp, sn) / batch_size_
            regterm = (torch.sign(batch_target_code) - batch_target_code).pow(2).sum() / (1e4 * batch_size_)
            loss_p = logloss + regterm
            loss_p.backward()
            pnet_optimizer.step()
            pnet_scheduler.step()
            set_requires_grad(pnet, False)

            x_adv = target_hash_adv(model, x, batch_target_code.sign(), epsilon, step=2, iteration=iteration,
                                    randomize=True)

            # optimize model
            model.zero_grad()

            adv_code = model(x_adv)
            U_ben[index, :] = ben_code.data

            ben_loss = model.loss_function(ben_code, y, index)

            S = cal_similarity(y, train_label)
            theta_x = adv_code.mm(U_ben.t()) / args.bit
            logloss = (theta_x - S.cuda()) ** 2
            adv_loss = 2 * logloss.sum() / (num_train * batch_size_)

            loss = ben_loss + adv_loss
            loss.backward()
            hash_optimizer.step()
            hash_scheduler.step()

            if it % 100 == 0:
                print('epoch: {:2d}, step: {:3d}, loss_p: {:.5f}, loss: {:.5f}, ben_loss: {:.5f}, adv_loss: {:.5f}'
                      .format(epoch, it, loss_p, loss, ben_loss, adv_loss))

    # torch.save(pnet, pnet_path)
    check_dir('log/atrdh_{}'.format(attack_model))
    robust_model_path = 'checkpoint/atrdh_{}.pth'.format(attack_model)
    torch.save(model, robust_model_path)
