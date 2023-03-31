import torch
import argparse
import utils.argument as argument
from model.attack_model.hag import hag
from model.attack_model.sdha import sdha
from model.attack_model.dhta import dhta
from model.attack_model.tha import tha
from adv_attack import adv_attack


torch.multiprocessing.set_sharing_strategy('file_system')


def parser_arguments():
    parser = argparse.ArgumentParser()
    
    parser = argument.add_base_arguments(parser)
    parser = argument.add_dataset_arguments(parser)
    parser = argument.add_model_arguments(parser)
    parser = argument.add_attack_arguments(parser)
    
    # arguments for defense
    parser.add_argument('--adv', dest='adv', action="store_true", default=False,
                        help='load model with adversarial training')
    parser = argument.add_defense_arguments(parser)

    # arguments for dataset
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='number of images in one batch')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()

    print("Current Method: {}".format(args.attack_method))
    if args.attack_method == 'hag':
        args.iteration = 2000
        hag(args)
    elif args.attack_method == 'sdha':
        args.iteration = 1500
        sdha(args, targeted=args.targeted)
    elif args.attack_method == 'dhta':
        args.iteration = 2000
        dhta(args)
    elif args.attack_method == 'p2p':
        args.iteration = 2000
        dhta(args, num_target=1)
    elif args.attack_method == 'tha':
        tha(args)
    elif args.attack_method == 'mainstay':
        adv_attack(args, targeted=args.targeted)
    else:
        raise NotImplementedError("Method {} not implemented".format(args.attack_method))
