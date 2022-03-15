import argparse
import torch.nn.functional as F
from tqdm import tqdm
from model.util import *
from utils.data_provider import *
from utils.hamming_matching import *
from utils.util import Logger, str2bool


torch.multiprocessing.set_sharing_strategy('file_system')


def adv_loss(noisy_output, target_hash):
    # loss = torch.mean(noisy_output * target_hash)
    sim = noisy_output * target_hash
    w = (sim > -0.5).int()
    sim = w * (sim + 2) * sim
    loss = torch.mean(sim)
    return loss


def hash_adv(model, query, target_hash, epsilon, step=1.0, iteration=100, randomize=False, record_loss=False):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    loss_list = [] if record_loss else None
    for i in range(iteration):
        alpha = get_alpha(i, iteration)
        noisy_output = model(query + delta, alpha)
        loss = adv_loss(noisy_output, target_hash.detach())
        loss.backward()

        # delta.data = delta - step * delta.grad.detach()
        # delta.data = delta - step * delta.grad.detach() / (torch.norm(delta.grad.detach(), 2) + 1e-9)
        delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

        if loss_list is not None and (i + 1) % (iteration // 10) == 0:
            loss_list.append(round(loss.item(), 4))
    if loss_list is not None:
        print("loss :{}".format(loss_list))
    return query + delta.detach()


def hash_center_code(label, train_code, train_label, bit):
    code = torch.zeros(label.size(0), bit).cuda()
    for i in range(label.size(0)):
        l = label[i].repeat(train_label.size(0), 1)  # N*C
        w = torch.sum(l * train_label, dim=1) / torch.sum(torch.sign(l + train_label), dim=1)  # N
        w1 = w.repeat(bit, 1).t()  # N*bit
        w2 = 1 - torch.sign(w1)  # N*bit, weights for negative samples
        c = w2.sum() / bit   # number of dissimilar samples
        w1 = 1 - w2  # weights for positive samples
        code[i] = torch.sign(torch.sum(c * w1 * train_code - (train_label.size(0) - c) * w2 * train_code, dim=0))
    return code


def central_attack(args, epsilon=8/255.):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    method = 'DHCA'
    # load model
    attack_model = get_attack_model_name(args)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                    args.batch_size, shuffle=False)
    train_loader, num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                              args.batch_size, shuffle=True)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    # load hashcode and labels
    database_code, _ = get_database_code(model, database_loader, attack_model)
    test_label = get_data_label(args.data_dir, args.dataset, 'test')
    database_label = get_data_label(args.data_dir, args.dataset, 'database')

    # generate hashcode and labels for training set
    train_code, train_label = generate_code(model, train_loader)
    train_code, train_label = torch.from_numpy(train_code), torch.from_numpy(train_label)
    train_code, train_label = train_code.cuda(), train_label.cuda()

    query_code_arr = np.zeros([num_test, args.bit], dtype=np.float32)
    adv_code_arr = np.zeros([num_test, args.bit], dtype=np.float32)
    center_code_arr = np.zeros([num_test, args.bit], dtype=np.float32)
    perceptibility = 0

    for it, data in enumerate(tqdm(test_loader, ncols=50)):
        query, label, index = data
        query, label = query.cuda(), label.cuda()
        batch_size_ = index.size(0)

        center_code = hash_center_code(label, train_code, train_label, args.bit)
        adv_query = hash_adv(model, query, center_code, epsilon, iteration=args.iteration)

        perceptibility += F.mse_loss(query, adv_query).data * batch_size_
        query_code = model(query).sign().cpu().data.numpy()
        adv_code = model(adv_query).sign().cpu().data.numpy()
        query_code_arr[index.numpy(), :] = query_code
        adv_code_arr[index.numpy(), :] = adv_code
        center_code_arr[index.numpy(), :] = center_code.cpu().data.numpy()

        if args.sample and it == 0:
            print("Sample images at iteration {}".format(it))
            sample_images(query[:16].cpu().numpy(), adv_query[:16].cpu().numpy(), attack_model, method=method, batch=it)

        if args.retrieve and it == 0:
            print("Retrieve images at iteration {}".format(it))
            # retrieve by original queries
            images_arr, labels_arr = retrieve_images(query.cpu().numpy(), label.cpu().numpy(), query_code,
                                                     database_code, 10, args.data_dir, args.dataset)
            save_retrieval_images(images_arr, labels_arr, 'ori', attack_model, it)
            images_arr, labels_arr = retrieve_images(adv_query.cpu().numpy(), label.cpu().numpy(), adv_code,
                                                     database_code, 10, args.data_dir, args.dataset)
            save_retrieval_images(images_arr, labels_arr, 'adv', attack_model, it)

    # save code
    np.save(os.path.join('log', attack_model, 'Original_code.npy'), query_code_arr)
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), adv_code_arr)

    # calculate map
    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    logger.log('perceptibility: {:.5f}'.format(torch.sqrt(perceptibility / num_test)))

    map_val = cal_map(database_code, adv_code_arr, database_label, test_label, 5000)
    logger.log('DHCA MAP(retrieval database): {:.5f}'.format(map_val))
    map_val = cal_map(database_code, -center_code_arr, database_label, test_label, 5000)
    logger.log('Theory MAP(retrieval database): {:.5f}'.format(map_val))
    map_val = cal_map(database_code, query_code_arr, database_label, test_label, 5000)
    logger.log('Ori MAP(retrieval database): {:.5f}'.format(map_val))


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', dest='method', default='hag', help='name of attack method')
    parser.add_argument('--dataset_name', dest='dataset', default='NUS-WIDE',
                        choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],
                        help='name of the dataset')
    parser.add_argument('--data_dir', dest='data_dir', default='../data/', help='path of the dataset')
    parser.add_argument('--device', dest='device', type=str, default='0', help='gpu device')
    parser.add_argument('--hash_method', dest='hash_method', default='DPH',
                        choices=['DPH', 'DPSH', 'HashNet'],
                        help='deep hashing methods')
    parser.add_argument('--backbone', dest='backbone', default='AlexNet',
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50', 'ResNet101'],
                        help='backbone network')
    parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of images in one batch')
    parser.add_argument('--iteration', dest='iteration', type=int, default=100, help='number of images in one batch')
    parser.add_argument('--retrieve', dest='retrieve', type=str2bool, default=False, help='retrieve images')
    parser.add_argument('--sample', dest='sample', type=str2bool, default=False, help='sample adversarial examples')
    parser.add_argument('--adv', dest='adv', type=str2bool, default='False',
                        help='load model after adversarial training')
    parser.add_argument('--adv_method', dest='adv_method', type=str, default='cat', choices=['cat', 'atrdh'],
                        help='adversarial training method')
    parser.add_argument('--lambda', dest='p_lambda', type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument('--mu', dest='p_mu', type=float, default=1e-4, help='mu for quantization loss')
    return parser.parse_args()


if __name__ == '__main__':
    central_attack(parser_arguments())
