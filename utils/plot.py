import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable


color_tuple = ('#8ECFC9', '#82B0D2', '#BEB8DC', '#FA7F6F', '#FFBE7A', '#E7DAD2',
               '#999999')

style_tuple = ('-', '--', '-.', ':')


def plot_curve(curve_arr, curve_label, color=None, style=None, curve_type='pr'):
    if not isinstance(curve_arr, Iterable):
        curve_arr = (curve_arr,)
        curve_label = (curve_label,)

    if color is None:
        color = np.random.choice(color_tuple, len(curve_label))

    if style is None:
        style = np.random.choice(style_tuple, len(curve_label))

    for i, curve in enumerate(curve_arr):
        x = curve[:, 0]
        y = curve[:, 1]
        plt.plot(x, y, label=curve_label[i], c=color[i], ls=style[i], lw=2)

    x_label = 'Recall' if curve_type == 'pr' else 'topN'
    plt.xlabel(x_label)
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    topn_arr = np.load('../log/NUS-WIDE_DPH_AlexNet_16/topn.npy')
    topn_label = np.loadtxt('../log/NUS-WIDE_DPH_AlexNet_16/topn.txt', dtype=str)
    plot_curve(topn_arr, topn_label, curve_type='topn')

    # pr_arr = np.load('../log/NUS-WIDE_DPH_AlexNet_32/pr.npy')
    # pr_label = np.loadtxt('../log/NUS-WIDE_DPH_AlexNet_32/pr.txt', dtype=str)
    # plot_curve(pr_arr, pr_label)
