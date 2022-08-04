import os
import torch
import argparse
from adv_attack import generate_mainstay_code
from model.util import load_model, generate_code_ordered
from utils.util import check_dir
from utils.data_provider import get_data_loader, get_classes_num


def adv_loss(adv_code, target_code):
    loss = torch.mean(adv_code * target_code)
    return loss


def adv_generator(model, query, target_hash, epsilon, step=2, iteration=7, alpha=1.0):
    delta = torch.zeros_like(query).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        adv_code = model(query + delta, alpha)
        loss = adv_loss(adv_code, target_hash.detach())
        loss.backward()

        delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

    return query + delta.detach()


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', dest='dataset', default='NUS-WIDE',
                        choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],
                        help='name of the dataset')
    parser.add_argument('--data_dir', dest='data_dir', default='../data/', help='path of the dataset')
    parser.add_argument('--device', dest='device', type=str, default='0', help='gpu device')
    parser.add_argument('--hash_method', dest='hash_method', default='DPH',
                        choices=['DPH', 'DPSH', 'HashNet', 'CSQ'],
                        help='deep hashing methods')
    parser.add_argument('--backbone', dest='backbone', default='AlexNet',
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50'],
                        help='backbone network')
    parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of images in one batch')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--iteration', dest='iteration', type=int, default=7, help='iteration of adversarial attack')
    parser.add_argument('--lambda', dest='p_lambda', type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument('--mu', dest='p_mu', type=float, default=1e-4, help='mu for quantization loss')
    return parser.parse_args()


def saat(args, epsilon=8 / 255.0):
    print("=> lambda: {}, mu: {}".format(args.p_lambda, args.p_mu))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    train_loader, num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                              args.batch_size, shuffle=True)

    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    print("Generating train code and label")
    train_code, train_label = generate_code_ordered(model, train_loader, num_train, args.bit,
                                                    get_classes_num(args.dataset))

    model.train()
    U_ben = torch.zeros(num_train, args.bit).cuda()
    U_ben.data = train_code.data

    # initialize `U` and `Y` of hashing model
    if hasattr(model, 'U') and hasattr(model, 'Y'):
        model.U.data = train_code.data
        model.Y.data = train_label.data

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5)
    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # adversarial training
    for epoch in range(args.epochs):
        epoch_loss = 0
        for it, (query, label, idx) in enumerate(train_loader):
            query, label = query.cuda(), label.cuda()

            # inner minimization aims to generate adversarial examples
            mainstay_code = generate_mainstay_code(label, U_ben, train_label)
            adv_query = adv_generator(model, query, mainstay_code, epsilon, step=2, iteration=args.iteration)

            # outer maximization aims to optimize parameters of model
            model.zero_grad()
            adv_code = model(adv_query)
            ben_code = model(query)
            U_ben[idx, :] = torch.sign(ben_code.data)
            mainstay_code = generate_mainstay_code(label, U_ben, train_label)

            loss_hash_ben = model.loss_function(ben_code, label, idx)
            loss_adv = - adv_loss(adv_code, mainstay_code)
            loss_qua = torch.mean((adv_code - torch.sign(adv_code)) ** 2)
            loss = args.p_lambda * loss_adv + args.p_mu * loss_qua + loss_hash_ben
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if it % 50 == 0:
                print("loss: {:.5f}\tben: {:.5f}\tadv: {:.5f}\tqua: {:.5f}".format(loss, loss_hash_ben.item(),
                                                                     loss_adv.item(), loss_qua.item()))

        print('Epoch: %3d/%3d\tTrain_loss: %3.5f \n' % (epoch, args.epochs, epoch_loss / len(train_loader)))

    if args.p_lambda != 1.0 or args.p_mu != 1e-4:
        robust_model = 'saat_{}_{}_{}'.format(attack_model, args.p_lambda, args.p_mu)
    else:
        robust_model = 'saat_{}'.format(attack_model)

    check_dir('log/{}'.format(robust_model))
    robust_model_path = 'checkpoint/{}.pth'.format(robust_model)
    torch.save(model, robust_model_path)


if __name__ == '__main__':
    saat(parser_arguments())
