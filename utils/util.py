import os
import argparse
import time


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


def check_dir(path, isdir=True):
     """
     Check whether the `path` is exist.
     isdir: `True` indicates the path is a directory, otherwise is a file.
     """
     path = '/'.join(path.split('/')[:-1]) if not isdir else path
     if not os.path.isdir(path):
         os.makedirs(path)


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
