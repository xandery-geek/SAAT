import os
import time
import torch
import numpy as np
from utils.data_provider import HashingDataset
from utils.hamming_matching import cal_hamming_dis


def load_model(path):
    print("Loading {}".format(path))
    model = torch.load(path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def get_attack_model_name(args):
    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    if args.adv:
        attack_model = '{}_{}'.format(args.adv_method, attack_model)
        if args.p_lambda != 1.0 or args.p_mu != 1e-4:
            attack_model = '{}_{}_{}'.format(attack_model, args.p_lambda, args.p_mu)
    return attack_model


def generate_code(model, data_loader):
    hash_code_list, labels_list = [], []
    model.eval()
    for images, labels, _ in data_loader:
        images = images.cuda()
        outputs = model(images)
        hash_code_list.append(outputs.data.cpu())
        labels_list.append(labels)
    return torch.cat(hash_code_list).sign().numpy(), torch.cat(labels_list).numpy()


def generate_code_ordered(model, data_loader, num_data, bit, num_class):
    code = torch.zeros([num_data, bit]).cuda()
    label = torch.zeros(num_data, num_class).cuda()
    for it, data in enumerate(data_loader, 0):
        data_input, data_label, data_ind = data
        output = model(data_input.cuda())
        code[data_ind, :] = torch.sign(output.data)
        label[data_ind, :] = data_label.cuda()
    return code, label


def get_database_code(model, dataloader, attack_model):
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    database_path = 'log/{}'.format(attack_model)
    database_hash_file = os.path.join(database_path, 'database_hashcode.npy')
    database_labels_file = os.path.join(database_path, 'database_label.npy')
    if os.path.exists(database_hash_file) and os.path.exists(database_labels_file):
        # check time stamp
        code_stamp = get_time_stamp(database_hash_file)
        label_stamp = get_time_stamp(database_labels_file)
        model_stamp = get_time_stamp(model_path)

        if model_stamp < code_stamp and model_stamp < label_stamp:
            print("Loading")
            print("hash code: {}".format(database_hash_file))
            print("label: {}".format(database_labels_file))

            database_code = np.load(database_hash_file)
            database_labels = np.load(database_labels_file)
            return database_code, database_labels

    print("Generating database code")
    database_code, database_labels = generate_code(model, dataloader)
    if not os.path.exists(database_path):
        os.makedirs(database_path)
    np.save(database_hash_file, database_code)
    np.save(database_labels_file, database_labels)
    return database_code, database_labels


def get_time_stamp(file):
    stamp = os.stat(file).st_mtime
    return time.localtime(stamp)


def get_alpha(cur_epoch, epochs):
    alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    epoch2alpha = epochs / len(alpha)
    return alpha[int(cur_epoch / epoch2alpha)]


def sample_images(ori_images, adv_images, attack_model, method, batch=0):
    np.save(os.path.join('log', attack_model, '{}-original-{}.npy'.format(method, batch)), np.array([ori_images]))
    np.save(os.path.join('log', attack_model, '{}-adversarial-{}.npy'.format(method, batch)), np.array([adv_images]))


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


def save_retrieval_images(images_arr, labels_arr, name, attack_model, iteration):
    np.save(os.path.join('log', attack_model, '{}_retrieve_images_{}.npy'.format(name, iteration)), images_arr)
    np.save(os.path.join('log', attack_model, '{}_retrieve_labels_{}.npy'.format(name, iteration)), labels_arr)
