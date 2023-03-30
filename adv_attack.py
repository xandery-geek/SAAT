import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model.util import get_attack_model_name, load_model, generate_code, get_database_code
from model.util import sample_images, retrieve_images, save_retrieval_images
from utils.data_provider import get_data_loader, get_data_label, get_classes_num
from utils.hamming_matching import cal_map, cal_perceptibility
from utils.util import Logger, import_class


def generate_mainstay_code(label, train_code, train_label):
    B = label.size(0)  # batch size
    N = train_label.size(0)  # number of training data

    w_1 = label @ train_label.t()
    label_norm = torch.norm(label, p=2, dim=1, keepdim=True).repeat(1, N)  # B * N
    train_label_norm = torch.norm(train_label, p=2, dim=1, keepdim=True).repeat(1, B) # N * B
    w_1 = w_1 / (label_norm * train_label_norm.t() + 1e-8)  # B * N
    w_2 = 1 - w_1.sign()

    n_p = 1 / torch.sum(w_1, dim=1, keepdim=True)
    w_1 = n_p.where(n_p != torch.inf, torch.tensor([0], dtype=torch.float).cuda()) * w_1

    n_n = 1 / torch.sum(w_2, dim=1, keepdim=True)
    w_2 = n_n.where(n_n != torch.inf, torch.tensor([0], dtype=torch.float).cuda()) * w_2

    code = torch.sign(w_1 @ train_code - w_2 @ train_code)  # B * K
    return code


def select_target_label(data_labels, unique_label):
    """
    select label which is different form original label
    :param data_labels: labels of original datas
    :param unique_label: candidate target labels
    :return: target label for targeted attack
    """
    # remove zero label
    target_label_sum = np.sum(unique_label, axis=1)
    zero_label_idx = np.where(target_label_sum == 0)[0]
    unique_label = np.delete(unique_label, zero_label_idx, axis=0)

    target_idx = []
    similarity = data_labels @ unique_label.transpose()
    for i, _ in enumerate(data_labels):
        s = similarity[i]
        candidate_idx = np.where(s == 0)[0]
        target_idx.append(np.random.choice(candidate_idx, size=1)[0])
    return unique_label[np.array(target_idx)]


def get_generator(name):
    return import_class('model.adv_generator.{}.{}Generator'.format(str.lower(name), name))


def adv_attack(args, epsilon=8 / 255., targeted=False, generator='PGD'):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    method = 'Ours' + ('_targeted' if targeted else '')
    # load model
    attack_model = get_attack_model_name(args)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, _ = get_data_loader(args.data_dir, args.dataset, 'database',
                                         args.batch_size, shuffle=False)
    train_loader, _ = get_data_loader(args.data_dir, args.dataset, 'train',
                                      args.batch_size, shuffle=True)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    # load hashcode and labels
    database_code, _ = get_database_code(model, database_loader, attack_model)
    test_label = get_data_label(args.data_dir, args.dataset, 'test')
    database_label = get_data_label(args.data_dir, args.dataset, 'database')

    # generate hashcode and labels for training set
    train_code, train_label = generate_code(model, train_loader)
    train_code, train_label = torch.from_numpy(train_code).cuda(), torch.from_numpy(train_label).cuda()

    # load target label for targeted attack
    if targeted:
        target_label_path = 'log/target_label_{}.txt'.format(args.dataset)
        if os.path.exists(target_label_path):
            target_label = np.loadtxt(target_label_path, dtype=np.int32)
        else:
            print("Generating target labels")
            unique_label = np.unique(database_label, axis=0)
            target_label = select_target_label(test_label, unique_label)
            np.savetxt(target_label_path, target_label, fmt="%d")
    else:
        target_label = test_label

    adv_generator = get_generator(generator)(model, epsilon, iteration=args.iteration, targeted=targeted)
    perceptibility = torch.tensor([0, 0, 0], dtype=torch.float)
    query_code_arr, adv_code_arr, mainstay_code_arr = None, None, None
    for it, (query, label, idx) in enumerate(tqdm(test_loader)):
        query, label = query.cuda(), label.cuda()
        batch_size_ = query.size(0)

        if targeted:
            if args.retrieve:
                category = 4
                target_l = torch.zeros((1, get_classes_num(args.dataset)))
                target_l[0, category] = 1
                target_l = target_l.repeat(batch_size_, 1).cuda()
            else:
                target_l = torch.from_numpy(target_label[idx]).cuda()
        else:
            target_l = label

        mainstay_code = generate_mainstay_code(target_l.float(), train_code, train_label.float())
        adv_query = adv_generator(query, mainstay_code)

        perceptibility += cal_perceptibility(query.cpu().detach(), adv_query.cpu().detach()) * batch_size_

        query_code = model(query).sign().cpu().detach().numpy()
        adv_code = model(adv_query).sign().cpu().detach().numpy()
        mainstay_code = mainstay_code.cpu().detach().numpy()

        query_code_arr = query_code if query_code_arr is None else np.concatenate((query_code_arr, query_code))
        adv_code_arr = adv_code if adv_code_arr is None else np.concatenate((adv_code_arr, adv_code))
        mainstay_code_arr = mainstay_code if mainstay_code_arr is None else np.concatenate((mainstay_code_arr, mainstay_code))

        if args.sample and it == 0:
            print("Sample images at iteration {}".format(it))
            sample_images(query[:16].cpu().numpy(), adv_query[:16].cpu().numpy(), attack_model, method=method, batch=it)

        if args.retrieve and it == 0:
            print("Retrieve images at iteration {}".format(it))
            # retrieve by original queries
            images_arr, labels_arr = retrieve_images(query.cpu().numpy(), label.cpu().numpy(), query_code,
                                                     database_code, 10, args.data_dir, args.dataset)
            save_retrieval_images(images_arr, labels_arr, 'ori', attack_model, it)
            # retrieve by adversarial queries
            images_arr, labels_arr = retrieve_images(adv_query.cpu().numpy(), label.cpu().numpy(), adv_code,
                                                     database_code, 10, args.data_dir, args.dataset)
            save_retrieval_images(images_arr, labels_arr, 'adv', attack_model, it)

    # save code
    np.save(os.path.join('log', attack_model, 'Original_code.npy'), query_code_arr)
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), adv_code_arr)

    # calculate map
    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    logger.log('perceptibility: {}'.format(perceptibility / num_test))

    map_val = cal_map(database_code, query_code_arr, database_label, test_label, 5000)
    logger.log('Ori MAP(retrieval database): {:.5f}'.format(map_val))

    if targeted:
        map_val = cal_map(database_code, adv_code_arr, database_label, target_label, 5000)
        logger.log('Ours t-MAP(retrieval database): {:.5f}'.format(map_val))
        map_val = cal_map(database_code, mainstay_code_arr, database_label, target_label, 5000)
        logger.log('Theory t-MAP(retrieval database): {:.5f}'.format(map_val))
    else:
        map_val = cal_map(database_code, adv_code_arr, database_label, test_label, 5000)
        logger.log('Ours MAP(retrieval database): {:.5f}'.format(map_val))
        map_val = cal_map(database_code, -mainstay_code_arr, database_label, test_label, 5000)
        logger.log('Theory MAP(retrieval database): {:.5f}'.format(map_val))


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targeted', dest='targeted', action="store_true", default=False, help='targeted attack')
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
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='number of images in one batch')
    parser.add_argument('--iteration', dest='iteration', type=int, default=100, help='number of training iteration')
    parser.add_argument('--retrieve', dest='retrieve', action="store_true", default=False, help='retrieve images')
    parser.add_argument('--sample', dest='sample', action="store_true", default=False,
                        help='sample adversarial examples')
    parser.add_argument('--adv', dest='adv', action="store_true", default=False,
                        help='load model after adversarial training')
    parser.add_argument('--adv_method', dest='adv_method', type=str, default='saat', choices=['saat', 'atrdh'],
                        help='adversarial training method')
    parser.add_argument('--lambda', dest='p_lambda', type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument('--mu', dest='p_mu', type=float, default=1e-4, help='mu for quantization loss')
    parser.add_argument('--generator', dest='generator', type=str, default='PGD', help='adversarial generator')
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser_arguments()
    adv_attack(args, targeted=args.targeted, generator=args.generator)
