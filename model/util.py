import os
import torch
import numpy as np


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


def get_database_code(model, dataloader, attack_model):
    database_path = 'log/{}'.format(attack_model)
    database_hash_file = os.path.join(database_path, 'database_hashcode.npy')
    database_labels_file = os.path.join(database_path, 'database_label.npy')
    if os.path.exists(database_hash_file) and os.path.exists(database_labels_file):
        database_code = np.load(database_hash_file)
        database_labels = np.load(database_labels_file)
    else:
        print("generate database code")
        database_code, database_labels = generate_code(model, dataloader)
        if not os.path.exists(database_path):
            os.makedirs(database_path)
        np.save(database_hash_file, database_code)
        np.save(database_labels_file, database_labels)
    return database_code, database_labels


def get_alpha(cur_epoch, epochs):
    alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    epoch2alpha = epochs / len(alpha)
    return alpha[int(cur_epoch / epoch2alpha)]


def save_images(ori_images, adv_images, attack_model, method, batch=0):
    np.save(os.path.join('log', attack_model, '{}-original-{}.npy'.format(method, batch)), np.array([ori_images]))
    np.save(os.path.join('log', attack_model, '{}-adversarial-{}.npy'.format(method, batch)), np.array([adv_images]))
