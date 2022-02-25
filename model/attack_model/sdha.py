# SDHA A Smart Adversarial Attack on Deep Hashing Based Image Retrieval

import torch.nn.functional
from tqdm import tqdm
from utils.hamming_matching import *
from model.attack_model.util import *
from utils.data_provider import get_data_loader, get_data_label
from utils.util import Logger


def load_optimizer(param):
    return torch.optim.Adam(param, lr=0.01, betas=(0.9, 0.999))


def cal_hamming_dis(b1, b2):
    """
    :param b1: k
    :param b2: n * k
    :return:
    """
    k = b2.size(1)
    return 0.5 * (k - b1 @ b2.transpose(0, 1))


def surrogate_function(h_hat, h, idx, sigma=5, z=0.5):
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


def sdha_loss(x_hat, x, h_hat, h, idx, alpha=25.):
    k = h.size(1)
    alpha = alpha / k
    mse_loss = torch.nn.functional.mse_loss(x_hat, x)
    surrogate_loss = surrogate_function(h_hat, h, idx)
    return mse_loss + alpha * surrogate_loss


def attack(model, x, idx, epochs=100, epsilon=0.039, record_loss=True):
    x = x.cuda()
    h = model(x)
    x, h = x.detach(), h.detach()

    x_hat = x.clone()
    x_hat.requires_grad_(True)
    optimizer = load_optimizer([x_hat])

    h_hat = None
    loss_list = [] if record_loss else None
    # perceptibility = 0
    for epoch in tqdm(range(epochs), ncols=50):
        optimizer.zero_grad()
        alpha = get_alpha(epoch, epochs)
        h_hat = model(x_hat, alpha)
        loss = sdha_loss(x_hat, x, h_hat, h, idx)
        loss.backward()
        optimizer.step()

        # perceptibility += torch.sqrt(torch.mean(torch.square(x.cpu() - x_hat.detach().cpu())))
        x_hat.data = torch.max(torch.min(x_hat.data, x + epsilon), x - epsilon)
        x_hat.data = torch.clamp(x_hat.data, min=0, max=1)  # subject to [0, 1]

        if loss_list is not None and (epoch + 1) % (epochs // 10) == 0:
            loss_list.append(round(loss.item(), 4))

    print("loss: {}".format(loss_list))
    return h.cpu().sign(), h_hat.detach().cpu().sign(), x_hat.detach().cpu()


g_code, g_similarity = torch.tensor(0), torch.tensor(0)


def sdha(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    method = 'SDHA'
    # load model
    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                    args.batch_size, shuffle=False)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    test_labels = get_data_label(args.data_dir, args.dataset, 'test')

    # generate global code and label
    global g_code, g_similarity
    g_code = torch.zeros(num_test, args.bit).float().cuda()
    label = torch.from_numpy(test_labels).float().cuda()
    g_similarity = label @ label.transpose(0, 1)  # n * n

    for x, _, idx in test_loader:
        g_code[idx] = model(x.cuda()).detach()

    # attack
    test_code, test_code_hat = None, None
    for i, (x, label, idx) in enumerate(test_loader):
        h, h_hat, x_hat = attack(model, x, idx, epochs=args.iteration)
        test_code = h.numpy() if test_code is None else np.concatenate((test_code, h.numpy()), axis=0)
        test_code_hat = h_hat.numpy() if test_code_hat is None else np.concatenate((test_code_hat, h_hat.numpy()),
                                                                                     axis=0)
        if i == 0:
            save_images(x[:4].cpu().numpy(), x_hat[:4].numpy(), attack_model, method=method, batch=i)

    database_code, database_labels = get_database_code(model, database_loader, attack_model)

    # save code
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), test_code_hat)

    # calculate map
    ori_map = cal_map(database_code, test_code, database_labels, test_labels, 5000)
    adv_map = cal_map(database_code, test_code_hat, database_labels, test_labels, 5000)

    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    logger.log('Ori MAP(retrieval database): {}'.format(ori_map))
    logger.log('SDHA MAP(retrieval database): {}'.format(adv_map))
