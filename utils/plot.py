import matplotlib.pyplot as plt
import numpy as np

color_tuple = ('#8ECFC9', '#82B0D2', '#BEB8DC', '#FA7F6F', '#FFBE7A', '#E7DAD2',
               '#999999')

style_tuple = ('-', '--', '-.', ':')


def plot_pr_curve(pr_arr, label, color=None, style=None):
    if not isinstance(pr_arr, (list, tuple)):
        pr_arr = (pr_arr,)
        label = (label,)

    if color is None:
        color = np.random.choice(color_tuple, len(label))

    if style is None:
        style = np.random.choice(style_tuple, len(label))

    for i, pr in enumerate(pr_arr):
        r = pr[:, 0]
        p = pr[:, 1]

        plt.plot(r, p, label=label[i], c=color[i], ls=style[i], lw=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def plot_topn_curve(top_n_arr, label, color=None, style=None):
    if not isinstance(top_n_arr, (list, tuple)):
        top_n_arr = (top_n_arr,)
        label = (label,)

    if color is None:
        color = np.random.choice(color_tuple, len(label))

    if style is None:
        style = np.random.choice(style_tuple, len(label))

    for i, pr in enumerate(top_n_arr):
        top_n = pr[:, 0]
        p = pr[:, 1]

        plt.plot(top_n, p, label=label[i], c=color[i], ls=style[i], lw=2)

    plt.xlabel('topN')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # arr1 = np.load('../log/CIFAR-10_DPH_AlexNet_32/HAG-pr_adv.npy')
    # arr2 = np.load('../log/CIFAR-10_DPH_AlexNet_32/HAG-pr_ori.npy')
    # plot_pr_curve((arr1, arr2), ('adv', 'ori'), style=('-', '--'),
    #               color=('#8ECFC9', '#82B0D2', '#BEB8DC', '#FA7F6F'))

    arr1 = np.load('../log/CIFAR-10_DPH_AlexNet_32/HAG-topn_ori.npy')
    arr2 = np.load('../log/CIFAR-10_DPH_AlexNet_32/HAG-topn_adv.npy')
    plot_topn_curve((arr1, arr2), ('adv', 'ori'), style=('-', '--'),
                    color=('#8ECFC9', '#82B0D2', '#BEB8DC', '#FA7F6F'))
