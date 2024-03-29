import argparse
import os
import numpy as np
from utils.hamming_matching import cal_pr, cal_top_n
from utils.data_provider import get_data_label


def cal_pr_or_topn(dataset, hash_method, backbone, bit, data_dir='../data', curve_type='pr', target=False, **kwargs):
    func = cal_pr if curve_type == 'pr' else cal_top_n

    if target:
        method_tuple = ('Original', 'P2P', 'DHTA', 'THA',  'ProS-GAN', 'Ours_targeted')
    else:
        method_tuple = ('Original', 'HAG', 'SDHA', 'Ours')

    attack_model = '{}_{}_{}_{}'.format(dataset, hash_method, backbone, bit)

    log_path = 'log/{}'.format(attack_model)
    database_hash_file = os.path.join(log_path, 'database_hashcode.npy')
    database_labels_file = os.path.join(log_path, 'database_label.npy')
    database_code = np.load(database_hash_file)
    database_labels = np.load(database_labels_file)
    test_labels = get_data_label(data_dir, dataset, 'test')
    if target:
        target_labels = np.loadtxt('log/target_label_{}.txt'.format(dataset))
    else:
        target_labels = test_labels

    curve_arr = []
    curve_label = []
    for method in method_tuple:
        method_file = os.path.join(log_path, '{}_code.npy'.format(method))
        if os.path.exists(method_file):
            print("method: {}".format(method))
            code = np.load(method_file)
            if method == 'Original':
                curve = func(database_code, code, database_labels, test_labels, **kwargs)
            else:
                curve = func(database_code, code, database_labels, target_labels, **kwargs)
            curve_arr.append(curve)
            curve_label.append(method)
        else:
            print("{} is not existed".format(method))
    filename = curve_type + ('_targeted' if target else '')
    np.save(os.path.join(log_path, '{}.npy'.format(filename)), np.array(curve_arr))
    np.savetxt(os.path.join(log_path, '{}.txt'.format(filename)), np.array(curve_label), fmt='%s')


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', dest='dataset', default='NUS-WIDE',
                        choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],
                        help='name of the dataset')
    parser.add_argument('--hash_method', dest='hash_method', default='DPH',
                        choices=['DPH', 'DPSH', 'HashNet', 'CSQ'],
                        help='deep hashing methods')
    parser.add_argument('--backbone', dest='backbone', default='AlexNet',
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50'],
                        help='backbone network')
    parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
    parser.add_argument('--type', dest='type', type=str, default='pr', help='curve type')
    parser.add_argument('--targeted', dest='targeted', action="store_true", default=False, help='targeted attack')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()
    if args.type == 'pr':
        kwargs = {'interval': 0.01}
    else:
        kwargs = {'top_n': range(100, 1001, 10)}
    cal_pr_or_topn(args.dataset, args.hash_method, args.backbone, args.bit,
                   curve_type=args.type, target=args.targeted, **kwargs)
