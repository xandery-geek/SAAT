import argparse
from torch.autograd import Variable
from utils.data_provider import *
from model.util import load_model, generate_code


def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    S = 2 * S - 1
    return S


def log_trick(x):
    lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(
        x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt


def pairwise_loss_updated(u, U, y, Y):
    alpha = 0.1
    similarity = (y @ Y.t() > 0).float()
    dot_product = alpha * u @ U.t()
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

    # weight
    S1 = mask_positive.float().sum()
    S0 = mask_negative.float().sum()
    S = S0 + S1
    exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
    exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

    loss = exp_loss.sum() / S
    return loss


def adv_loss(noisy_output, target_hash):
    # loss = torch.mean(noisy_output * target_hash)
    sim = noisy_output * target_hash
    w = (sim > -0.5).int()
    sim = w * (sim + 2) * sim
    loss = torch.mean(sim)
    return loss


def hash_adv(model, query, target_hash, epsilon, step=2, iteration=7, randomize=True):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        noisy_output = model(query + delta)
        loss = adv_loss(noisy_output, target_hash.detach())
        loss.backward()

        delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

    return query + delta.detach()


def hash_center_code(y, B, L, bit):
    code = torch.zeros(y.size(0), bit).cuda()
    for i in range(y.size(0)):
        l = y[i].repeat(L.size(0), 1)
        w = torch.sum(l * L, dim=1) / torch.sum(torch.sign(l + L), dim=1)
        w1 = w.repeat(bit, 1).t()
        w2 = 1 - torch.sign(w1)
        c = w2.sum() / bit
        w1 = 1 - w2
        code[i] = torch.sign(torch.sum(c * w1 * B - (L.size(0) - c) * w2 * B, dim=0))
    return code


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
    parser.add_argument('--method', dest='method', default='hag', help='name of attack method')
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
    parser.add_argument('--lambda', dest='p_lambda', type=float, default=1, help='lambda for adversarial loss')
    parser.add_argument('--mu', dest='p_mu', type=float, default=1e-4, help='mu for quantization loss')
    return parser.parse_args()


def central_adv_train(args, epsilon=8 / 255.0, epochs=10, iteration=7):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    train_loader, num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                              args.batch_size, shuffle=True)

    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    robust_model_path = 'checkpoint/cat_{}.pth'.format(attack_model)
    model = load_model(model_path)

    train_B, train_L = generate_code(model, train_loader)
    train_B, train_L = torch.from_numpy(train_B), torch.from_numpy(train_L)
    train_B, train_L = train_B.cuda(), train_L.cuda()

    U_ben = torch.zeros(num_train, args.bit).cuda()
    U_ben.data = train_B.data

    U_adv = torch.zeros(num_train, args.bit).cuda()
    B_ben = torch.zeros(num_train, args.bit).cuda()

    B_ben.data = train_B.data

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5)
    lr_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # adversarial training
    for epoch in range(epochs):
        epoch_loss = 0
        for it, data in enumerate(train_loader):
            x, y, index = data
            x = x.cuda()
            y = y.cuda()

            center_codes_curr = hash_center_code(y, B_ben, train_L, args.bit)
            x_adv = hash_adv(model, x, center_codes_curr, epsilon, step=2, iteration=iteration, randomize=True)
            x_adv = x_adv.detach()

            model.zero_grad()
            # mixup
            # x_adv = mixup(x, x_adv)
            output_adv = model(x_adv)
            # y_ben = model.classify()
            U_adv[index, :] = output_adv.data
            output_ben = model(x)
            # y_adv = model.classify()
            B_ben[index, :] = torch.sign(output_ben.data)
            U_ben[index, :] = output_ben.data

            # DPH
            if args.hash_method == 'DPH':
                S = CalcSim(y, train_L)
                theta_x = output_ben.mm(U_ben.t()) / args.bit
                loss_hash_ben = torch.mean((theta_x - S) ** 2)
            elif args.hash_method == 'DPSH':
                S = CalcSim(y, train_L)
                S = (S + 1) / 2
                omega = output_ben.mm(U_ben.t()) / 2
                loss_hash_ben = torch.mean(-(S * omega - log_trick(omega)))
                Bbatch = torch.sign(output_ben.data)
                quan = torch.mean((output_ben - Bbatch)**2)
                loss_hash_ben += 1e-4*quan
            elif args.hash_method == 'DPSH':
                loss_hash_ben = pairwise_loss_updated(output_ben, U_ben, y, train_L)
            else:
                raise NotImplementedError()

            center_codes = hash_center_code(y, B_ben, train_L, args.bit)

            loss_adv = - torch.mean((output_adv * center_codes))
            loss_qua = torch.mean((output_adv - torch.sign(output_adv)) ** 2)
            loss = args.p_lambda * loss_adv + args.p_mu * loss_qua + loss_hash_ben
            loss.backward()
            opt.step()
            scheduler.step()
            epoch_loss += loss.item()

            if it % 50 == 0:
                print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, loss: {:.5f}'.format(epoch, it, scheduler.get_last_lr()[0],
                                                                                   loss))

        print('Epoch: %3d/%3d\tTrain_loss: %3.5f \n' % (epoch, epochs, epoch_loss / len(train_loader)))
    torch.save(model, robust_model_path)


if __name__ == '__main__':
    central_adv_train(parser_arguments())
