import argparse
from model.attack_model.hag import hag
from model.attack_model.sdha import sdha


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', default='hag', help='name of attack model')
    parser.add_argument('--dataset_name', dest='dataset', default='NUS-WIDE',
                        choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],
                        help='name of the dataset')
    parser.add_argument('--data_dir', dest='data_dir', default='../data/', help='path of the dataset')
    parser.add_argument('--device', dest='device', type=str, default='0', help='gpu device')
    parser.add_argument('--hash_method', dest='hash_method', default='DPH',
                        choices=['DPH', 'DPSH', 'HashNet'],
                        help='deep hashing methods')
    parser.add_argument('--backbone', dest='backbone', default='AlexNet',
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50'],
                        help='backbone network')
    parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help='number of images in one batch')
    parser.add_argument('--iteration', dest='iteration', type=int, default=2000, help='number of images in one batch')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()

    if args.model == 'hag':
        hag(args)
    elif args.model == 'sdha':
        sdha(args)
