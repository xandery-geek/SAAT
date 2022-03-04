import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from collections.abc import Iterable


color_tuple = ('#7f7f7f', '#d8383a', '#8d69b8', '#519e3e',
               '#84584e', '#d57dbf', '#ef8636', '#3b75af')


style_tuple = ('-', '-', '-', '-', '-', '-', '-', '-')
marker_tuple = ('s', '^', '^', '^', '^', '*', '*', 'h')
ms_tuple = (6, 6, 6, 6, 6, 8, 8, 6)


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

    fig, ax = plt.subplots()
    plt.figure(1)

    for i, curve in enumerate(curve_arr):
        x = curve[:, 0]
        y = curve[:, 1]
        plt.plot(x, y, label=curve_label[i], c=color[i], ls=style[i], lw=1.5
                 , marker=marker_tuple[i], markersize=ms_tuple[i])

    x_major_locator = MultipleLocator(0.1 if curve_type == 'pr' else 100)
    y_major_locator = MultipleLocator(0.1)
    loc = (0.02, 0.43) if curve_type == 'pr' else (0.02, 0.37)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlabel('Recall' if curve_type == 'pr' else 'Number of top ranked samples')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc=loc, framealpha=0.7)
    plt.grid(linestyle='--')
    fig.subplots_adjust(left=0.09, right=0.99)
    fig.set_size_inches(6, 5.5)
    plt.savefig('../documents/{}-{}.svg'.format(title, curve_type), dpi=600, format='svg', transparent=True)
    # plt.show()


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
