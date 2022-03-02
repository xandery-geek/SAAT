import os
import argparse
import time

import numpy as np
from utils.data_provider import HashingDataset
from utils.hamming_matching import cal_hamming_dis


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])

    # return mod.comp1.comp2...
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_batch(data_loader, batch):
    # get data batch
    it = iter(data_loader)
    i = 0
    while i < batch:
        it.next()
        i += 1
    return it.next()


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def retrieve_images(query_images, query_labels, query_codes, database_codes, top, data_dir, dataset):
    # calculate top index
    retrieve_indices = []
    for query in query_codes:
        hamming_dis = cal_hamming_dis(query, database_codes)
        sort_index = np.argsort(hamming_dis)
        retrieve_indices.append(sort_index[:top])

    # get top images and labels
    database = HashingDataset(os.path.join(data_dir, dataset), 'database_img.txt', 'database_label.txt')

    batch_images_arr, batch_labels_arr = [], []
    for i, indices in enumerate(retrieve_indices):
        # query images and labels
        images_arr, labels_arr = [query_images[i]], [query_labels[i]]
        # retrieve images and labels
        for index in indices:
            image, label, _ = database[index]
            images_arr.append(image.numpy())
            labels_arr.append(label.numpy())
        batch_images_arr.append(images_arr)
        batch_labels_arr.append(labels_arr)

    return np.array(batch_images_arr), np.array(batch_labels_arr)


class Logger(object):
    def __init__(self, path, filename):
        self.log_file = os.path.join(path, filename)

    def log(self, string, print_time=True):
        if print_time:
            localtime = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
            string = "[" + localtime + '] ' + string
        print(string)
        with open(self.log_file, 'a') as f:
            print(string, file=f)
