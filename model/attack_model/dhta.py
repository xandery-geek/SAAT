# DHTA: Targeted Attack for Deep Hashing based Retrieval

import os
import torch
import numpy as np
from utils.data_provider import get_data_loader, get_data_label
from utils.hamming_matching import cal_map
from model.util import get_alpha, get_attack_model_name, load_model, get_database_code
from utils.util import Logger
from tqdm import tqdm


def adv_loss(adv_code, target_code):
    loss = -torch.mean(adv_code * target_code)
    return loss


def adv_generator(model, query, target_code, epsilon, step=1, iteration=2000, randomize=False):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        alpha = get_alpha(i, iteration)
        adv_code = model(query + delta, alpha)
        loss = adv_loss(adv_code, target_code)
        loss.backward()

        delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()
    return query + delta.detach()


def generate_anchor_code(hash_codes):
    return torch.sign(torch.sum(hash_codes, dim=0))


def dhta(args, num_target=9, epsilon=0.032):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    method = 'P2P' if num_target == 1 else 'DHTA'

    # load model
    attack_model = get_attack_model_name(args)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                    args.batch_size, shuffle=False)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    database_code, _ = get_database_code(model, database_loader, attack_model)

    test_label = get_data_label(args.data_dir, args.dataset, 'test')
    database_label = get_data_label(args.data_dir, args.dataset, 'database')

    # convert one-hot code to string
    database_labels_str = [''.join(label) for label in database_label.astype(str)]
    database_labels_str = np.array(database_labels_str, dtype=str)

    target_label_path = 'log/target_label_{}.txt'.format(args.dataset)
    if os.path.exists(target_label_path):
        print("Loading target label from {}".format(target_label_path))
        target_labels = np.loadtxt(target_label_path, dtype=np.int)
    else:
        raise ValueError('Please generate target_label before attack!')

    target_labels_str = [''.join(label) for label in target_labels.astype(str)]

    adv_code_arr = np.zeros([num_test, args.bit], dtype=np.float32)
    anchor_code_arr = np.zeros((num_test, args.bit), dtype=np.float)
    for it, data in enumerate(tqdm(test_loader, ncols=50)):
        query, _, idx = data
        query = query.cuda()
        batch_size_ = idx.size(0)

        batch_anchor_code = torch.zeros((batch_size_, args.bit), dtype=torch.float)
        for i in range(batch_size_):
            # select hash code which has the same label with target from database randomly
            target_label_str = target_labels_str[idx[0] + i]
            anchor_idx = np.where(database_labels_str == target_label_str)
            anchor_idx = np.random.choice(anchor_idx[0], size=num_target)

            anchor_code = generate_anchor_code(torch.from_numpy(database_code[anchor_idx]))
            anchor_code = anchor_code.view(1, args.bit)
            batch_anchor_code[i, :] = anchor_code

        anchor_code_arr[it * args.batch_size:it * args.batch_size + batch_size_] = batch_anchor_code.numpy()
        adv_query = adv_generator(model, query, batch_anchor_code.cuda(), epsilon, iteration=args.iteration)
        u_ind = np.linspace(it * args.batch_size, np.min((num_test, (it + 1) * args.batch_size)) - 1,
                            batch_size_, dtype=int)

        adv_code = model(adv_query).sign().cpu().data.numpy()
        adv_code_arr[u_ind, :] = adv_code

    # save code
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), adv_code_arr)

    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    t_map = cal_map(database_code, anchor_code_arr, database_label, target_labels, 5000)
    logger.log('AnchorCode t-MAP(retrieval database) :{:.5f}'.format(t_map))
    t_map = cal_map(database_code, adv_code_arr, database_label, target_labels, 5000)
    logger.log('{} t-MAP(retrieval database) :{:.5f}'.format(method, t_map))
