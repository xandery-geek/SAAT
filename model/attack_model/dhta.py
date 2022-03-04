import collections
import pandas as pd
from utils.data_provider import *
from utils.hamming_matching import *
from model.util import *
from utils.util import Logger
from tqdm import tqdm


def target_adv_loss(noisy_output, target_hash):
    loss = -torch.mean(noisy_output * target_hash)
    return loss


def target_hash_adv(model, query, target_hash, epsilon, step=1, iteration=2000, randomize=False):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        alpha = get_alpha(i, iteration)
        noisy_output = model(query + delta, alpha)
        # noisy_output = model(query + delta)
        loss = target_adv_loss(noisy_output, target_hash)
        loss.backward()

        # delta.data = delta - step * delta.grad.detach() / (torch.norm(delta.grad.detach(), 2) + 1e-9)
        delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
        # delta.data = delta - step * delta.grad.detach()
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

        # if i % 10 == 0:
        #     print('it:{}, loss:{}'.format(i, loss))

    # print(torch.min(255*delta.data))
    # print(torch.max(255*delta.data))
    return query + delta.detach()


def generate_hash(model, samples, num_data, bit):
    output = model(samples)
    B = torch.sign(output.cpu().data).numpy()
    return B


def hash_anchor_code(hash_codes):
    return torch.sign(torch.sum(hash_codes, dim=0))


def sample_image(image, name, sample_dir='sample/dhta'):
    image = image.cpu().detach()[0]
    image = transforms.ToPILImage()(image.float())
    image.save(os.path.join(sample_dir, name + '.png'), quality=100)


def dhta(args, num_target=9, epsilon=0.032):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    method = 'P2P' if num_target == 1 else 'DHTA'

    # load model
    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    if args.atrdh:
        attack_model = 'atrdh_{}'.format(attack_model)
    else:
        attack_model = attack_model if not args.adv else 'cat_{}'.format(attack_model)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                    args.batch_size, shuffle=False)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    database_hash, _ = get_database_code(model, database_loader, attack_model)

    test_labels_int = get_data_label(args.data_dir, args.dataset, 'test')
    database_labels_int = get_data_label(args.data_dir, args.dataset, 'database')

    # convert one-hot code to string
    database_labels_str = [''.join(label) for label in database_labels_int.astype(str)]
    database_labels_str = np.array(database_labels_str, dtype=str)

    # calculate target label
    target_label_path = 'log/target_label_dhta_{}.txt'.format(args.dataset)

    if os.path.exists(target_label_path):
        print("Loading target label from {}".format(target_label_path))
        target_labels = np.loadtxt(target_label_path, dtype=np.int)
    else:
        print("Generating target label")
        candidate_labels_count = collections.Counter(database_labels_str)
        candidate_labels_count = pd.DataFrame.from_dict(candidate_labels_count, orient='index').reset_index()
        candidate_labels = candidate_labels_count[candidate_labels_count[0] > num_target]['index']
        candidate_labels = np.array(candidate_labels, dtype=str)

        candidate_labels_int = [list(candidate_labels[i]) for i in range(len(candidate_labels))]
        candidate_labels_int = np.array(candidate_labels_int, dtype=np.int)
        # print(candidate_labels_int.shape)

        target_labels = []
        S = np.dot(test_labels_int, candidate_labels_int.T)
        for i in range(num_test):
            label_ori = test_labels_int[i]
            s = S[i]
            candidate_index = np.where(s == 0)
            random_index = np.random.choice(candidate_index[0])
            target_label = candidate_labels_int[random_index]
            target_label = np.array(target_label, dtype=int)
            target_labels.append(target_label)

        target_labels = np.array(target_labels, dtype=np.int)
        np.savetxt(target_label_path, target_labels, fmt="%d")

    target_labels_str = [''.join(label) for label in target_labels.astype(str)]

    qB = np.zeros([num_test, args.bit], dtype=np.float32)
    query_anchor_codes = np.zeros((num_test, args.bit), dtype=np.float)
    # perceptibility = 0
    for it, data in enumerate(tqdm(test_loader, ncols=50)):
        queries, _, index = data
        # sample_image(queries, '{}_benign'.format(it))

        queries = queries.cuda()
        batch_size_ = index.size(0)

        anchor_codes = torch.zeros((batch_size_, args.bit), dtype=torch.float)
        for i in range(batch_size_):
            # select hash code which has the same label with target from database randomly
            target_label_str = target_labels_str[index[0] + i]
            anchor_indexes = np.where(database_labels_str == target_label_str)
            anchor_indexes = np.random.choice(anchor_indexes[0], size=num_target)

            anchor_code = hash_anchor_code(torch.from_numpy(database_hash[anchor_indexes]))
            anchor_code = anchor_code.view(1, args.bit)
            anchor_codes[i, :] = anchor_code

        query_anchor_codes[it * args.batch_size:it * args.batch_size + batch_size_] = anchor_codes.numpy()
        query_adv = target_hash_adv(model, queries, anchor_codes.cuda(), epsilon, iteration=args.iteration)
        u_ind = np.linspace(it * args.batch_size, np.min((num_test, (it + 1) * args.batch_size)) - 1,
                            batch_size_, dtype=int)

        # perceptibility += F.mse_loss(queries, query_adv).data * batch_size_
        query_code = generate_hash(model, query_adv, batch_size_, args.bit)
        qB[u_ind, :] = query_code

    # save code
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), qB)

    a_map = cal_map(database_hash, query_anchor_codes, database_labels_int, target_labels, 5000)
    a_t_map = cal_map(database_hash, query_anchor_codes, database_labels_int, test_labels_int, 5000)
    t_map = cal_map(database_hash, qB, database_labels_int, target_labels, 5000)
    _map = cal_map(database_hash, qB, database_labels_int, test_labels_int, 5000)

    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    logger.log('AnchorCode t-MAP(retrieval database) :{}'.format(a_map))
    logger.log('AnchorCode MAP(retrieval database) :{}'.format(a_t_map))
    logger.log('{} t-MAP(retrieval database) :{}'.format(method, t_map))
    logger.log('{} MAP(retrieval database): {}'.format(method, _map))
