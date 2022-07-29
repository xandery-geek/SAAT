import argparse
import torch.nn.functional as F
from tqdm import tqdm
from model.util import *
from utils.data_provider import *
from utils.hamming_matching import *
from utils.util import Logger, str2bool

torch.multiprocessing.set_sharing_strategy('file_system')


def adv_loss(noisy_output, target_code):
    # loss = torch.mean(noisy_output * target_code)
    sim = noisy_output * target_code
    w = (sim > -0.5).int()
    m = w.sum()
    sim = w * (sim + 2) * sim
    loss = sim.sum()/m
    return loss

def adv_loss_targeted(noisy_output, target_code):
    sim = noisy_output * target_code
    w = (sim < 0.5).int()
    m = w.sum()
    sim = w * (sim + 2) * sim
    loss = -sim.sum()/m
    return loss


def hash_adv(model, query, target_code, epsilon, step=1.0, iteration=100, targeted=False, record_loss=False):
    # random initialization
    delta = torch.zeros_like(query).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    loss_func = adv_loss_targeted if targeted else adv_loss
    loss_list = [] if record_loss else None
    for i in range(iteration):
        alpha = get_alpha(i, iteration)
        noisy_output = model(query + delta, alpha)
        loss = loss_func(noisy_output, target_code.detach())
        loss.backward()

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
    B = label.size(0)  # batch size
    N = train_label.size(0)  # number of training data
    C = label.size(1)  # number of classes

    # w_1 = (label @ train_label.t() > 0).float()
    # w_2 = 1 - w_1
    
    w_1 = (label @ train_label.t())/torch.sum(label, dim=1, keepdim=True)  # B * N
    mask_p = w_1.sign()  # mask of positives

    label_sum = torch.sum(label, dim=1, keepdim=True).repeat(1, N)  # B * N
    train_label_sum = torch.sum(train_label, dim=1, keepdim=True).repeat(1, B)  # N * B
    w_2 = (label_sum + train_label_sum.t())*(1 - mask_p)/C

    w_p = 1 / torch.sum(w_1, dim=1, keepdim=True)
    w_1 = w_p.where(w_p != torch.inf, torch.tensor([0], dtype=torch.float).cuda()) * w_1

    w_n = 1 / torch.sum(w_2, dim=1, keepdim=True)
    w_2 = w_n.where(w_n != torch.inf, torch.tensor([0], dtype=torch.float).cuda()) * w_2
    
    code = torch.sign(w_1 @ train_code - w_2 @ train_code)  # B * K
    return code


def central_attack(args, epsilon=8 / 255., targeted=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    method = 'DHCA'
    # load model
    attack_model = get_attack_model_name(args)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, _ = get_data_loader(args.data_dir, args.dataset, 'database',
                                         args.batch_size, shuffle=False)
    train_loader, _ = get_data_loader(args.data_dir, args.dataset, 'train',
                                      args.batch_size, shuffle=True)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    # load hashcode and labels
    database_code, _ = get_database_code(model, database_loader, attack_model)
    test_label = get_data_label(args.data_dir, args.dataset, 'test')
    database_label = get_data_label(args.data_dir, args.dataset, 'database')

    # generate hashcode and labels for training set
    train_code, train_label = generate_code(model, train_loader)
    train_code, train_label = torch.from_numpy(train_code).cuda(), torch.from_numpy(train_label).cuda()

    # load target label for targeted attack
    if targeted:
        target_label_path = 'log/target_label_{}.txt'.format(args.dataset)
        if os.path.exists(target_label_path):
            target_label = np.loadtxt(target_label_path, dtype=np.int)
        else:
            raise ValueError('Please generate target_label before attack!')

    perceptibility = 0
    query_code_arr, adv_code_arr, center_code_arr = None, None, None
    for it, (query, label, idx) in enumerate(tqdm(test_loader, ncols=50)):
        query, label = query.cuda(), label.cuda()
        batch_size_ = query.size(0)

        if not targeted:
            target_l = label
        else:
            if args.retrieve:
                category = 4
                target_l = torch.zeros((1, get_classes_num(args.dataset)))
                target_l[0, category] = 1
                target_l = target_l.repeat(batch_size_, 1).cuda()
            else:
                target_l = torch.from_numpy(target_label[idx]).cuda()

        center_code = hash_center_code(target_l.float(), train_code, train_label.float(), args.bit)
        adv_query = hash_adv(model, query, center_code, epsilon, iteration=args.iteration, targeted=targeted)

        perceptibility += F.mse_loss(query, adv_query).data * batch_size_

        query_code = model(query).sign().cpu().detach().numpy()
        adv_code = model(adv_query).sign().cpu().detach().numpy()
        center_code = center_code.cpu().detach().numpy()

        query_code_arr = query_code if query_code_arr is None else np.concatenate((query_code_arr, query_code), axis=0)
        adv_code_arr = adv_code if adv_code_arr is None else np.concatenate((adv_code_arr, adv_code), axis=0)
        center_code_arr = center_code if center_code_arr is None else np.concatenate((center_code_arr, center_code),
                                                                                     axis=0)

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

    map_val = cal_map(database_code, query_code_arr, database_label, test_label, 5000)
    logger.log('Ori MAP(retrieval database): {:.5f}'.format(map_val))

    if not targeted:
        map_val = cal_map(database_code, adv_code_arr, database_label, test_label, 5000)
        logger.log('DHCA MAP(retrieval database): {:.5f}'.format(map_val))
        map_val = cal_map(database_code, -center_code_arr, database_label, test_label, 5000)
        logger.log('Theory MAP(retrieval database): {:.5f}'.format(map_val))
    else:
        map_val = cal_map(database_code, adv_code_arr, database_label, target_label, 5000)
        logger.log('DHCA t-MAP(retrieval database): {:.5f}'.format(map_val))
        map_val = cal_map(database_code, center_code_arr, database_label, target_label, 5000)
        logger.log('Theory t-MAP(retrieval database): {:.5f}'.format(map_val))


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
