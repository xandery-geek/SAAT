import os
import time
import torch
import argparse
import numpy as np
import utils.argument as argument
from tqdm import tqdm
from utils.data_provider import get_data_loader, get_classes_num
from utils.util import import_class, get_batch
from utils.hamming_matching import cal_map
from model.util import retrieve_images


torch.multiprocessing.set_sharing_strategy('file_system')


class Hashing(object):
    def __init__(self, args):
        self.args = args

        # load dataset
        self.database_loader, self.num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                                  args.batch_size, shuffle=False)
        self.train_loader, self.num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                                            args.batch_size, shuffle=True)
        self.test_loader, self.num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                                          args.batch_size, shuffle=False)

        # load model
        if args.load:
            self.model = self.load_model()
        else:
            model = 'model.hash_model.{}.{}'.format(str.lower(args.hash_method), args.hash_method)
            num_class = get_classes_num(args.dataset)
            self.model = import_class(model)(dataset=args.dataset,
                                             **{'bit': args.bit,
                                                'backbone': args.backbone,
                                                'num_train': self.num_train,
                                                'num_class': num_class})
            if torch.cuda.is_available():
                self.model = self.model.cuda()

        # load optimizer
        if args.train:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.args.lr,
                                             momentum=self.args.momentum,
                                             weight_decay=self.args.wd)
            self.loss_function = self.model.loss_function

        self.model_name = self.model.model_name
        self.log_dir = os.path.join('log', self.model_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.print_log("Model: {}".format(self.model_name))

    def load_model(self):
        model_path = '{}/{}_{}_{}_{}.pth'.format(self.args.save, self.args.dataset, self.args.hash_method,
                                                 self.args.backbone, self.args.bit)
        self.print_log("Loading Model: {}".format(model_path))
        model = torch.load(model_path)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def print_log(self, string, print_time=True):
        if print_time:
            localtime = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
            string = "[" + localtime + '] ' + string
        print(string)
        with open('{}/log.txt'.format(self.log_dir), 'a') as f:
            print(string, file=f)

    def _train(self, epoch):
        self.print_log('Train Epoch {}:'.format(epoch))
        batch_number = len(self.train_loader)
        avg_loss = 0
        process = tqdm(self.train_loader)
        for i, (inputs, labels, index) in enumerate(process):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels.float(), index)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()

        avg_loss /= batch_number
        self.print_log("loss: {:.5f}".format(avg_loss))
        return avg_loss

    def adjust_learning_rate(self, epoch, total_epoch):
        lr = self.args.lr * (0.1 ** (epoch // (total_epoch // 3)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_dataset(self, split):
        if split == 'train':
            data_loader = self.train_loader
        elif split == 'test':
            data_loader = self.test_loader
        elif split == 'database':
            data_loader = self.database_loader
        else:
            raise ValueError("Unknown dataset {}".format(split))
        return data_loader

    def generate_code(self, split):
        data_loader = self.get_dataset(split)
        hash_code_list, labels_list = [], []

        self.model.eval()
        for images, labels, _ in tqdm(data_loader):
            images = images.cuda()
            outputs = self.model(images)
            hash_code_list.append(outputs.data.cpu())
            labels_list.append(labels)
        return torch.cat(hash_code_list).sign().numpy(), torch.cat(labels_list).numpy()

    def load_database(self):
        hash_code_arr = np.load(os.path.join(self.log_dir, 'database_hashcode.npy'))
        labels_arr = np.load(os.path.join(self.log_dir, 'database_label.npy'))
        return hash_code_arr, labels_arr

    def train(self):
        self.print_log('>>>Training Model<<<')
        self.model.train()
        record_loss = []
        for epoch in range(0, self.args.n_epochs):
            loss = self._train(epoch)
            record_loss.append(loss)
            self.adjust_learning_rate(epoch, self.args.n_epochs)
        torch.save(self.model, os.path.join(self.args.save, self.model_name + '.pth'))
        self.test()

    def test(self):
        self.print_log('>>>Testing MAP<<<')
        if not args.load:
            self.model = self.load_model()
        test_hash_codes, test_labels = self.generate_code('test')
        retrieval_hash_codes, retrieval_labels = self.generate_code('database')

        map_val = cal_map(retrieval_hash_codes, test_hash_codes, retrieval_labels, test_labels, 5000)
        self.print_log("Test MAP: {:.5f}".format(map_val))

    def retrieve(self, batch=0, top=10):
        self.print_log('>>>Retrieve relevant images<<<')
        if not args.load:
            self.model = self.load_model()
        self.model.eval()

        # get data batch
        images, labels, _ = get_batch(self.test_loader, batch)
        # calculate hash code
        outputs = self.model(images.cuda())
        outputs = outputs.data.cpu()
        database_codes, _ = self.load_database()

        images_arr, labels_arr = retrieve_images(images.numpy(), labels.numpy(), outputs, database_codes, top,
                                                 args.data_dir, args.dataset)

        print("Writing retrieve images of database to {}".format(self.log_dir))
        np.save(os.path.join(self.log_dir, 'retrieve_images_{}.npy'.format(batch)), images_arr)
        np.save(os.path.join(self.log_dir, 'retrieve_labels_{}.npy'.format(batch)), labels_arr)

    def generate(self):
        self.print_log('>>>Generating hash code<<<')
        if not args.load:
            self.model = self.load_model()
        hash_code_arr, labels_arr = self.generate_code('database')
        hash_code_arr, labels_arr = hash_code_arr.astype(int), labels_arr.astype(int)

        print("Writing hash code of database to {}".format(self.log_dir))
        np.save(os.path.join(self.log_dir, 'database_hashcode.npy'), hash_code_arr)
        np.save(os.path.join(self.log_dir, 'database_label.npy'), labels_arr)
        with open(os.path.join(self.log_dir, 'database.txt'), 'w') as f:
            for i in range(len(hash_code_arr)):
                print('{},{},{}'.format(i, ' '.join(map(str, hash_code_arr[i])), ' '.join(map(str, labels_arr[i]))),
                      file=f)


def parser_arguments():
    parser = argparse.ArgumentParser()
    
    parser = argument.add_base_arguments(parser)
    parser = argument.add_dataset_arguments(parser)
    parser = argument.add_model_arguments(parser)

    # arguments for different phases
    parser.add_argument('--train', dest='train', action='store_true', default=False, help='to train or not')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='to test or not')
    parser.add_argument('--generate', dest='generate', action='store_true', default=False, help='to generate or not')
    parser.add_argument('--retrieve', dest='retrieve', action='store_true', default=False, help='to retrieve or not')

    # arguments for training
    parser.add_argument('--load_model', dest='load', action='store_true', default=False, help='load the latest model for continue training')
    parser.add_argument('--checkpoint_dir', dest='save', default='checkpoint', help='models are saved here')
    parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=100, help='number of epoch')
    parser.add_argument('--learning_rate', dest='lr', type=float, default=0.01, help='initial learning rate for SGD')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', dest='wd', type=float, default=5e-4, help='weight decay for SGD')

    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of images in one batch')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    hashing = Hashing(args)
    if args.train:
        hashing.train()
    if args.test:
        hashing.test()
    if args.generate:
        hashing.generate()
    if args.retrieve:
        hashing.retrieve()
