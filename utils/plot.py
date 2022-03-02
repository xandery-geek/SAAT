import argparse

import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable

# color_tuple = ('#8ECFC9', '#82B0D2', '#BEB8DC', '#FA7F6F', '#FFBE7A', '#E7DAD2',
#                '#999999', '#00fbff')

color_tuple = ('#d8383a', '#8d69b8', '#d57dbf', '#FA7F6F',
               '#7f7f7f', '#519e3e', '#ef8636', '#3b75af')

# color_tuple = ('#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2',
#                '#BEB8DC', '#E7DAD2', '#999999', '#96C37D')

style_tuple = ('-', '-', '-', '-', '-', '-', '-', '-')
marker_tuple = ('o', 'o', 'o', 'o', 'o', 'o', 'o', 'o')
ms_tuple = (4, 4, 4, 4, 4, 4, 4, 4)


def plot_curve(curve_arr, curve_label, title='', color=None, style=None, curve_type='pr'):
    if not isinstance(curve_arr, Iterable):
        curve_arr = (curve_arr,)
        curve_label = (curve_label,)

    if color is None:
        # color = np.random.choice(color_tuple, len(curve_label))
        color = color_tuple

    if style is None:
        # style = np.random.choice(style_tuple, len(curve_label))
        style = style_tuple

    for i, curve in enumerate(curve_arr):
        x = curve[:, 0]
        y = curve[:, 1]
        plt.plot(x, y, label=curve_label[i], c=color[i], ls=style[i], lw=1.5
                 , marker=marker_tuple[i], markersize=ms_tuple[i])

    x_label = 'Recall' if curve_type == 'pr' else 'Number of top ranked samples'
    loc = (0.02, 0.4) if curve_type == 'pr' else (0.7, 0.4)
    plt.xlabel(x_label)
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc=loc)
    plt.show()


def parser_arguments():
    parser = argparse.ArgumentParser()
    # description of data
    parser.add_argument('--dataset_name', dest='dataset', default='NUS-WIDE',
                        choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],
                        help='name of the dataset')
    parser.add_argument('--hash_method', dest='hash_method', default='DPH',
                        choices=['DPH', 'DPSH', 'HashNet'],
                        help='deep hashing methods')
    parser.add_argument('--backbone', dest='backbone', default='AlexNet',
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50'],
                        help='backbone network')
    parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
    parser.add_argument('--type', dest='type', type=str, default='pr', help='curve type')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()
    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    arr = np.load('../log/{}/{}.npy'.format(attack_model, args.type))
    label = np.loadtxt('../log/{}/{}.txt'.format(attack_model, args.type), dtype=str)
    plot_curve(arr, label, title=args.dataset, curve_type=args.type)
