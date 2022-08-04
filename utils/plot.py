import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from collections.abc import Iterable

color_tuple = ('#7f7f7f', '#2ca02c', '#e377c2', '#1f77b4')

color_tuple_targeted = ('#7f7f7f', '#2ca02c', '#e377c2', '#9467bd', '#8c564b','#1f77b4')

style_tuple = ('-', '-', '-', '-', '-', '-', '-', '-')
marker_tuple = ('s', '^', '^', '^', '^', '*', '*', 'h')
ms_tuple = (8, 9, 9, 9, 9, 9, 9, 9)


def plot_curve(curve_arr, curve_label, title='', color=None, style=None, curve_type='pr', targeted=False):
    filter_label = ('Original',) if targeted else ('',)
    if not isinstance(curve_arr, Iterable):
        curve_arr = (curve_arr,)
        curve_label = (curve_label,)

    if color is None:
        color = color_tuple_targeted if targeted else color_tuple

    if style is None:
        style = style_tuple

    fig, ax = plt.subplots()
    plt.figure(1)

    for i, curve in enumerate(curve_arr):
        if curve_label[i] in filter_label:
            continue
        x = curve[:, 0]
        y = curve[:, 1]
        plt.plot(x, y, label=curve_label[i], c=color[i], ls=style[i], lw=2.5)

    x_major_locator = MultipleLocator(0.2 if curve_type == 'pr' else 200)
    y_major_locator = MultipleLocator(0.1)
    loc = (0.05, 0.05) if targeted else (0.05, 0.65)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0.05, 0.7)
    plt.xlabel('Recall' if curve_type == 'pr' else 'Number of top ranked samples')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc=loc, framealpha=0.7)
    # plt.grid(linestyle='--')
    fig.subplots_adjust(left=0.09, right=0.99)
    fig.set_size_inches(6, 5.5)
    filename = curve_type + ('_targeted' if targeted else '')
    plt.savefig('../documents/{}-{}.pdf'.format(title, filename), dpi=600, format='pdf', transparent=True)
    # plt.show()


def plot_ablation(row, para_type='lambda'):
    import pandas as pd
    data = pd.DataFrame(pd.read_excel('../documents/AttackMAP.xlsx', sheet_name=0))
    curve_label = ['Original', 'HAG', 'SDHA', 'Ours']

    parameters = []
    map_value = []
    for i in range(row[0], row[1]):
        value = data.loc[i].values
        if i == row[0]:
            parameters = [float(v) for v in value[1:7]]
        else:
            map_value.append([float(v) for v in value[1:7]])
    curve_arr = np.array([list(zip(parameters, val)) for val in map_value])

    color, style = color_tuple, style_tuple
    fig, ax = plt.subplots()
    plt.figure(1)

    for i, curve in enumerate(curve_arr):
        x = range(len(curve[:, 0]))
        y = curve[:, 1]
        plt.plot(x, y, label=curve_label[i], c=color[i], ls=style[i], lw=2.5, markerfacecolor='none'
                 , markeredgewidth=2, marker=marker_tuple[i], markersize=ms_tuple[i])
    plt.xticks(range(curve_arr.shape[1]), labels=[str(i) for i in curve_arr[0, :, 0]])
    plt.xlabel('weighting factor ' + ('$\lambda$' if para_type == 'lambda' else '$\mu$'))
    plt.ylabel('MAP(%)')
    plt.legend(loc=(0.05, 0.65))
    plt.grid(linestyle='--')
    fig.subplots_adjust(left=0.1, right=0.99)
    plt.savefig('../documents/{}-{}.pdf'.format('ablation', para_type), dpi=600, format='pdf', transparent=True)
    # plt.show()


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
    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    filename = args.type + ('_targeted' if args.targeted else '')
    arr = np.load('../log/{}/{}.npy'.format(attack_model, filename))
    label = np.loadtxt('../log/{}/{}.txt'.format(attack_model, filename), dtype=str)
    label = [l if 'Ours' not in l else 'Ours' for l in label]
    plot_curve(arr, label, title=args.dataset, curve_type=args.type, targeted=args.targeted)
    # plot_ablation(row=(89, 94), para_type='lambda')
    # plot_ablation(row=(97, 102), para_type='mu')

