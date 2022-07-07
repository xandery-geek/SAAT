import argparse
from model.defense_model.atrdh import atrdh
from central_adv_train import dhcat


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', dest='method', default='dhcat', choices=['dhcat', 'atrdh'],
                        help='name of defense method')
    parser.add_argument('--dataset_name', dest='dataset', default='NUS-WIDE',
                        choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],
                        help='name of the dataset')
    parser.add_argument('--data_dir', dest='data_dir', default='../data/', help='path of the dataset')
    parser.add_argument('--device', dest='device', type=str, default='0', help='gpu device')
    parser.add_argument('--hash_method', dest='hash_method', default='DPH', choices=['DPH', 'DPSH', 'CSQ', 'HashNet'],
                        help='deep hashing methods')
    parser.add_argument('--backbone', dest='backbone', default='AlexNet',
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50', 'ResNet101'],
                        help='backbone network')
    parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of images in one batch')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--iteration', dest='iteration', type=int, default=7, help='iteration of adversarial attack')
    parser.add_argument('--lambda', dest='p_lambda', type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument('--mu', dest='p_mu', type=float, default=1e-4, help='mu for quantization loss')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()
    print("Current Defense Method: {}".format(args.method))
    if args.method == 'dhcat':
        dhcat(args)
    elif args.method == 'atrdh':
        args.epochs = 100
        atrdh(args)
    else:
        raise NotImplementedError("Method {} not implemented".format(args.method))
