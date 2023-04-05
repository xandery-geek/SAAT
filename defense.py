import argparse
import utils.argument as argument
from model.defense_model.atrdh import atrdh
from adv_training import saat


def parser_arguments():
    parser = argparse.ArgumentParser()
    
    parser = argument.add_base_arguments(parser)
    parser = argument.add_dataset_arguments(parser)
    parser = argument.add_model_arguments(parser)

    # arguments for defense
    parser = argument.add_defense_arguments(parser)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--iteration', dest='iteration', type=int, default=7, help='iteration of adversarial attack')

    # arguments for dataset
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of images in one batch')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()
    print("Current Defense Method: {}".format(args.adv_method))
    if args.adv_method == 'saat':
        saat(args)
    elif args.adv_method == 'atrdh':
        args.epochs = 100
        atrdh(args)
    else:
        raise NotImplementedError("Method {} not implemented".format(args.adv_method))
