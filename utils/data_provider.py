import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class HashingDataset(Dataset):
    def __init__(
            self,
            data_path,
            img_filename,
            label_filename,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).float()
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


def load_model(path):
    model = torch.load(path)
    model = model.cuda()
    model.eval()
    return model


def load_label(filename, data_dir):
    label_filepath = os.path.join(data_dir, filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    return torch.from_numpy(label).float()


def get_classes_num(dataset):
    classes_dic = {'FLICKR-25K': 38, 'NUS-WIDE': 21, 'MS-COCO': 80}
    return classes_dic[dataset]


def get_dataset_filename(split):
    filename = {
        'train': ('train_img.txt', 'train_label.txt'),
        'test': ('test_img.txt', 'test_label.txt'),
        'database': ('database_img.txt', 'database_label.txt')
    }
    return filename[split]


def get_data_loader(data_dir, dataset_name, split, batch_size, shuffle=False, num_workers=4):
    """
    return dataloader and data number
    :param num_workers:
    :param shuffle:
    :param batch_size:
    :param data_dir:
    :param dataset_name:
    :param split: choice from ('train, 'test', 'database')
    :return:
    """
    file_name, label_name = get_dataset_filename(split)
    dataset = HashingDataset(os.path.join(data_dir, dataset_name), file_name, label_name)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader, len(dataset)


def get_data_label(data_dir, dataset_name, split):
    _, label_name = get_dataset_filename(split)
    return np.loadtxt(os.path.join(data_dir, dataset_name, label_name), dtype=int)
