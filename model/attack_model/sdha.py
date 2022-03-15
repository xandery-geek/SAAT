# SDHA A Smart Adversarial Attack on Deep Hashing Based Image Retrieval

import torch.nn.functional
from tqdm import tqdm
from utils.hamming_matching import *
from model.util import *
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


def attack(model, x, idx, epochs=100, epsilon=8/255., record_loss=False):
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
        loss = sdha_loss(x_hat, x, h_hat, h, idx)
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


g_code, g_similarity = torch.tensor(0), torch.tensor(0)


def sdha(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    method = 'SDHA'
    # load model
    attack_model = get_attack_model_name(args)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                    args.batch_size, shuffle=False)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    test_label = get_data_label(args.data_dir, args.dataset, 'test')

    # generate global code and label
    global g_code, g_similarity
    g_code = torch.zeros(num_test, args.bit).float().cuda()
    label = torch.from_numpy(test_label).float().cuda()
    g_similarity = label @ label.transpose(0, 1)  # n * n

    for x, _, idx in test_loader:
        g_code[idx] = model(x.cuda()).detach()

    # attack
    query_code_arr, adv_code_arr = None, None
    for i, (x, label, idx) in enumerate(tqdm(test_loader, ncols=50)):
        h, h_hat, _ = attack(model, x, idx, epochs=args.iteration)
        # h, h_hat = theory_attack(model, x, idx)
        query_code_arr = h.numpy() if query_code_arr is None else np.concatenate((query_code_arr, h.numpy()), axis=0)
        adv_code_arr = h_hat.numpy() if adv_code_arr is None else np.concatenate((adv_code_arr, h_hat.numpy()), axis=0)

    database_code, database_label = get_database_code(model, database_loader, attack_model)

    # save code
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), adv_code_arr)

    # calculate map
    ori_map = cal_map(database_code, query_code_arr, database_label, test_label, 5000)
    adv_map = cal_map(database_code, adv_code_arr, database_label, test_label, 5000)

    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    logger.log('SDHA MAP(retrieval database): {:.5f}'.format(adv_map))
    logger.log('Ori MAP(retrieval database): {:.5f}'.format(ori_map))
