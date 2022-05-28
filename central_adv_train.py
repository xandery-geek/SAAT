import os
import torch
import argparse
from central_attack import hash_center_code
from model.util import load_model, generate_code_ordered
from utils.util import check_dir
from utils.data_provider import get_data_loader, get_classes_num


def adv_loss(noisy_output, target_hash):
    loss = torch.mean(noisy_output * target_hash)
    return loss


def hash_adv(model, query, target_hash, epsilon, step=2, iteration=7, randomize=True):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        alpha = 1.0
        noisy_output = model(query + delta, alpha)
        loss = adv_loss(noisy_output, target_hash.detach())
        loss.backward()

        delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

    return query + delta.detach()


def mixup(x, x_adv, gamma=2):
    batch_size = x.size(0)
    delta = x_adv - x
    delta = gamma * delta
    x_adv = (x + delta).clamp(-1, 1)
    alpha = torch.rand(batch_size)
    alpha = alpha.view(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(x).cuda()
    x_adv = alpha * x + (1 - alpha) * x_adv
    x_adv = x_adv.detach()
    return x_adv


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
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'],
                        help='backbone network')
    parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of images in one batch')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='epoch of adversarial training')
    parser.add_argument('--iteration', dest='iteration', type=int, default=7, help='iteration of adversarial attack')
    parser.add_argument('--lambda', dest='p_lambda', type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument('--mu', dest='p_mu', type=float, default=1e-4, help='mu for quantization loss')
    return parser.parse_args()


def dhcat(args, epsilon=8 / 255.0):
    print("Current lambda: {}, mu: {}".format(args.p_lambda, args.p_mu))

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

    if hasattr(model, 'U') and hasattr(model, 'Y'):
        model.U.data = train_code.data
        model.Y.data = train_label.data

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5)
    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # adversarial training
    for epoch in range(args.epochs):
        epoch_loss = 0
        for it, data in enumerate(train_loader):
            x, y, index = data
            x, y = x.cuda(), y.cuda()

            center_code = hash_center_code(y, U_ben, train_label, args.bit)
            x_adv = hash_adv(model, x, center_code, epsilon, step=2, iteration=args.iteration, randomize=True)
            x_adv = x_adv.detach()

            model.zero_grad()
            # mixup
            # x_adv = mixup(x, x_adv)
            adv_code = model(x_adv, 1.0)
            ben_code = model(x, 1.0)
            U_ben[index, :] = torch.sign(ben_code.data)
            center_code = hash_center_code(y, U_ben, train_label, args.bit)

            loss_hash_ben = model.loss_function(ben_code, y, index)
            loss_adv = - adv_loss(adv_code, center_code)
            loss_qua = torch.mean((adv_code - torch.sign(adv_code)) ** 2)
            loss = args.p_lambda * loss_adv + args.p_mu * loss_qua + loss_hash_ben
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if it % 50 == 0:
                print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, loss: {:.5f}'.format(
                        epoch, it, scheduler.get_last_lr()[0], loss))
                print("ben: {:.5f}, adv: {:.5f}, qua: {:.5f}".format(loss_hash_ben.item(),
                                                                     loss_adv.item(), loss_qua.item()))

        print('Epoch: %3d/%3d\tTrain_loss: %3.5f \n' % (epoch, args.epochs, epoch_loss / len(train_loader)))

    if args.p_lambda != 1.0 or args.p_mu != 1e-4:
        robust_model = 'cat_{}_{}_{}'.format(attack_model, args.p_lambda, args.p_mu)
    else:
        robust_model = 'cat_{}'.format(attack_model)

    check_dir('log/{}'.format(robust_model))
    robust_model_path = 'checkpoint/{}.pth'.format(robust_model)
    torch.save(model, robust_model_path)


if __name__ == '__main__':
    dhcat(parser_arguments())
