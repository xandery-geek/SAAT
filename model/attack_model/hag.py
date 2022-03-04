# HAG

import torch.nn.functional
from tqdm import tqdm
from utils.hamming_matching import *
from model.util import *
from utils.data_provider import get_data_loader, get_data_label
from utils.util import Logger


def load_optimizer(param):
    return torch.optim.SGD(param, lr=100, momentum=0.9)


def mask_code(adv_code, ori_code, threshold_t):
    """
    calculate mask code
    @adv_code: code of adversarial example
    @ori_code: code of original image
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
        loss = hag_loss(h_hat, h)
        loss.backward()
        optimizer.step()

        # perceptibility += torch.sqrt(torch.mean(torch.square(x.cpu() - x_hat.detach().cpu())))
        x_hat.data = torch.max(torch.min(x_hat.data, x + epsilon), x - epsilon)
        x_hat.data = torch.clamp(x_hat.data, min=0, max=1)  # subject to [0, 1]

        if loss_list is not None and (epoch + 1) % (epochs // 10) == 0:
            loss_list.append(round(loss.item(), 4))

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
    test_code, test_code_hat = None, None
    for i, (x, label, idx) in enumerate(test_loader):
        h, h_hat, x_hat = attack(model, x, idx, epochs=args.iteration)
        # h, h_hat, x_hat = theory_attack(model, x)
        test_code = h.numpy() if test_code is None else np.concatenate((test_code, h.numpy()), axis=0)
        test_code_hat = h_hat.numpy() if test_code_hat is None else np.concatenate((test_code_hat, h_hat.numpy()),
                                                                                   axis=0)
        # if i == 0:
        #     save_images(x[:4].cpu().numpy(), x_hat[:4].numpy(), attack_model, method=method, batch=i)

    database_code, database_labels = get_database_code(model, database_loader, attack_model)

    # save code
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), test_code_hat)

    # calculate map
    theory_map = cal_map(database_code, -test_code, database_labels, test_labels, 5000)
    ori_map = cal_map(database_code, test_code, database_labels, test_labels, 5000)
    adv_map = cal_map(database_code, test_code_hat, database_labels, test_labels, 5000)

    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    logger.log('Ori MAP(retrieval database): {}'.format(ori_map))
    logger.log('Theory MAP(retrieval database): {}'.format(theory_map))
    logger.log('HAG MAP(retrieval database): {}'.format(adv_map))

    # calculate P-R curve
    # pr_arr = cal_pr(database_code, test_code, database_labels, test_labels, interval=0.01)
    # np.save(os.path.join('log', attack_model, '{}-pr_ori.npy'.format(method)), pr_arr)
    #
    # pr_arr = cal_pr(database_code, test_code_hat, database_labels, test_labels, interval=0.01)
    # np.save(os.path.join('log', attack_model, '{}-pr_adv.npy'.format(method)), pr_arr)
    #
    # top_n = cal_top_n(database_code, test_code, database_labels, test_labels)
    # np.save(os.path.join('log', attack_model, '{}-topn_ori.npy'.format(method)), top_n)
    # top_n = cal_top_n(database_code, test_code_hat, database_labels, test_labels)
    # np.save(os.path.join('log', attack_model, '{}-topn_adv.npy'.format(method)), top_n)
