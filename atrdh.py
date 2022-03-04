import argparse
import os
import torch
import numpy as np
from torch.autograd import Variable
from utils.data_provider import get_data_loader, get_classes_num, get_data_label
from model.util import load_model
from utils.util import check_dir
from model.attack_model.tha import PrototypeNet, CircleLoss, similarity
from central_adv_train import CalcSim, log_trick, pairwise_loss_updated, generate_code_label


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

        # if i % 1 == 0:
        #     print('it:{}, loss:{}'.format(i, loss))
    return query + delta.detach()


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', dest='dataset', default='NUS-WIDE',
                        choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],
                        help='name of the dataset')
    parser.add_argument('--data_dir', dest='data_dir', default='../data/', help='path of the dataset')
    parser.add_argument('--device', dest='device', type=str, default='0', help='gpu device')
    parser.add_argument('--hash_method', dest='hash_method', default='DPH',
                        choices=['DPH', 'DPSH', 'HashNet'],
                        help='deep hashing methods')
    parser.add_argument('--backbone', dest='backbone', default='AlexNet',
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50', 'ResNet101'],
                        help='backbone network')
    parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of images in one batch')
    return parser.parse_args()


def atrdh(args, epsilon=8 / 255.0, epochs=100, iteration=7):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    num_classes = get_classes_num(args.dataset)
    train_loader, num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                              args.batch_size, shuffle=True)

    database_labels = get_data_label(args.data_dir, args.dataset, 'database')
    database_labels = torch.from_numpy(database_labels).float()
    target_labels = database_labels.unique(dim=0)

    pnet = PrototypeNet(args.bit, num_classes).cuda()
    pnet.train()
    model.train()
    circle_loss = CircleLoss(m=0, gamma=1)
    optimizer_l = torch.optim.Adam(pnet.parameters(), lr=1e-4, betas=(0.5, 0.999))
    # opt = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    lr_steps = epochs * len(train_loader)
    scheduler_l = torch.optim.lr_scheduler.MultiStepLR(optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                       gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    train_B, train_L = generate_code_label(model, train_loader, num_train, args.bit, get_classes_num(args.dataset))

    U_ben = torch.zeros(num_train, args.bit).cuda()
    U_adv = torch.zeros(num_train, args.bit).cuda()

    # adversarial training
    for epoch in range(epochs):
        epoch_loss = 0.0
        for it, data in enumerate(train_loader):
            x, y, index = data
            x = x.cuda()
            y = y.cuda()
            batch_size_ = index.size(0)

            output_ben = model(x)
            train_B[index.numpy(), :] = torch.sign(output_ben.detach())

            select_index = np.random.choice(range(target_labels.size(0)), size=batch_size_)
            batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()
            set_requires_grad(pnet, True)
            optimizer_l.zero_grad()
            target_hash_l = pnet(batch_target_label)
            sp, sn = similarity(target_hash_l, train_B, batch_target_label, train_L, args.bit)
            logloss = circle_loss(sp, sn) / (batch_size_)
            regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size_)
            loss_p = logloss + regterm
            loss_p.backward()
            optimizer_l.step()
            scheduler_l.step()

            batch_prototype_codes = target_hash_l
            prototype_codes = torch.sign(batch_prototype_codes)
            x_adv = target_hash_adv(model, x, prototype_codes, epsilon, step=2, iteration=iteration, randomize=True)

            set_requires_grad(pnet, False)
            model.zero_grad()
            output_adv = model(x_adv)
            for i, ind in enumerate(index):
                U_ben[ind, :] = output_ben.data[i]
                U_adv[ind, :] = output_adv.data[i]

            S = CalcSim(y, train_L)
            if args.hash_method == 'DPH':
                theta_x = output_ben.mm(Variable(U_ben).t()) / args.bit
                logloss = (theta_x - S.cuda()) ** 2
                loss = logloss.sum() / (num_train * batch_size_)
            elif args.hash_method == 'DPSH':
                S1 = (y.mm(train_L.t()) > 0).float()
                Bbatch = torch.sign(output_ben)
                theta_x = output_ben.mm(Variable(U_ben.cuda()).t()) / 2
                logloss = (Variable(S1.cuda()) * theta_x - log_trick(theta_x)).sum() / (num_train * batch_size_)
                regterm = (Bbatch - output_ben).pow(2).sum() / (num_train * batch_size_)
                loss = -logloss + 50 * regterm
            elif args.hash_method == 'HashNet':
                loss = pairwise_loss_updated(output_ben, U_ben.cuda(), y, train_L)
            else:
                raise NotImplementedError()

            theta_x = output_adv.mm(Variable(U_ben).t()) / args.bit
            logloss = (theta_x - S.cuda()) ** 2
            loss += 2 * logloss.sum() / (num_train * batch_size_)
            loss.backward()
            opt.step()
            scheduler.step()

            if it % 100 == 0:
                print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, loss_p: {:.5f}, loss: {:.5f}'.format(
                    epoch, it, scheduler.get_last_lr()[0], loss_p, loss))

    # torch.save(pnet, pnet_path)
    check_dir('log/atrdh_{}'.format(attack_model))
    robust_model_path = 'checkpoint/atrdh_{}.pth'.format(attack_model)
    torch.save(model, robust_model_path)


if __name__ == '__main__':
    atrdh(parser_arguments())
