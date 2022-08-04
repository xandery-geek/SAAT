import torch
import argparse
from model.attack_model.hag import hag
from model.attack_model.sdha import sdha
from model.attack_model.dhta import dhta
from model.attack_model.tha import tha
from adv_attack import adv_attack


torch.multiprocessing.set_sharing_strategy('file_system')


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', dest='method', default='mainstay', help='name of attack method')
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()

    print("Current Method: {}".format(args.method))
    if args.method == 'hag':
        args.iteration = 2000
        hag(args)
    elif args.method == 'sdha':
        args.iteration = 1500
        sdha(args, targeted=args.targeted)
    elif args.method == 'dhta':
        args.iteration = 2000
        dhta(args)
    elif args.method == 'p2p':
        args.iteration = 2000
        dhta(args, num_target=1)
    elif args.method == 'tha':
        tha(args)
    elif args.method == 'mainstay':
        adv_attack(args, targeted=args.targeted)
    else:
        raise NotImplementedError("Method {} not implemented".format(args.method))
