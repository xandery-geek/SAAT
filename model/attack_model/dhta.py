import time
import collections
import pandas as pd
import torch.nn.functional as F
from utils.data_provider import *
from utils.hamming_matching import *
from model.attack_model.util import *


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
        delta.data = delta - step/255 * torch.sign(delta.grad.detach())
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

# def sample_image(image, name, sample_dir='sample/dhta'):
#     image = image.cpu().numpy()[0] * 255
#     image = np.array(image, dtype=np.uint8)
#     image = np.transpose(image, (1,2,0))
#     image = Image.fromarray(image)
#     image.save(os.path.join(sample_dir, name + '.jpg'))


def dhta(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    DATABASE_FILE = 'database_img.txt'
    TEST_FILE = 'test_img.txt'
    DATABASE_LABEL = 'database_label.txt'
    TEST_LABEL = 'test_label.txt'

    epsilon = 8
    epsilon = epsilon / 255.
    n_t = 9
    iteration = 0
    method = 'DHTA'
    if n_t == 1:
        method = 'P2P'
    transfer = False

    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)
    database_code_path = 'log/database_code_{}_{}_{}_{}.txt'.format(dataset, model_name, backbone, bit)

    if transfer:
        t_model_name = 'DPH'
        t_bit = 32
        t_backbone = 'AlexNet'
        t_model_path = 'checkpoint/{}_{}_{}_{}.pth'.format(args.dataset, t_model_name, t_backbone, t_bit)
        t_model = load_model(t_model_path)
    else:
        t_model_name = args.hash_method
        t_bit = args.bit
        t_backbone = args.backbone
    t_database_code_path = 'log/database_code_{}_{}_{}_{}.txt'.format(dataset, t_model_name, t_backbone, t_bit)
    target_label_path = 'log/target_label_dhta_{}.txt'.format(dataset)
    test_code_path = 'log/test_code_{}_{}_{}.txt'.format(dataset, method, t_bit)


    # data processing
    database_loader, num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                    args.batch_size, shuffle=False)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    if os.path.exists(database_code_path):
        database_hash = np.loadtxt(database_code_path, dtype=np.float)
    else:
        database_hash = GenerateCode(model, database_loader, num_database, bit)
        np.savetxt(database_code_path, database_hash, fmt="%d")

    if os.path.exists(t_database_code_path):
        t_database_hash = np.loadtxt(t_database_code_path, dtype=np.float)
    else:
        t_database_hash = GenerateCode(t_model, database_loader, num_database, t_bit)
        np.savetxt(t_database_code_path, t_database_hash, fmt="%d")



    print('database hash codes prepared!')

    test_labels_int = np.loadtxt(os.path.join(DATA_DIR, TEST_LABEL), dtype=int)
    database_labels_int = np.loadtxt(os.path.join(DATA_DIR, DATABASE_LABEL), dtype=int)
    test_labels_str = [''.join(label) for label in test_labels_int.astype(str)]
    database_labels_str = [''.join(label) for label in database_labels_int.astype(str)]
    test_labels_str = np.array(test_labels_str, dtype=str)
    database_labels_str = np.array(database_labels_str, dtype=str)


    # target_labels = torch.from_numpy(database_labels_int).unique(dim=0)
    # print(target_labels.shape)



    if os.path.exists(target_label_path):
        target_labels = np.loadtxt(target_label_path, dtype=np.int)
    else:
        candidate_labels_count = collections.Counter(database_labels_str)
        candidate_labels_count = pd.DataFrame.from_dict(candidate_labels_count, orient='index').reset_index()
        candidate_labels = candidate_labels_count[candidate_labels_count[0] > n_t]['index']
        candidate_labels = np.array(candidate_labels, dtype=str)

        candidate_labels_int = [list(candidate_labels[i]) for i in range(len(candidate_labels))]
        candidate_labels_int = np.array(candidate_labels_int, dtype=np.int)
        # print(candidate_labels_int.shape)

        target_labels = []
        S = np.dot(test_labels_int, candidate_labels_int.T)
        for i in range(num_test):
            label_ori = test_labels_int[i]
            s = S[i]
            candidate_index = np.where(s==0)
            random_index = np.random.choice(candidate_index[0])
            target_label = candidate_labels_int[random_index]
            target_label = np.array(target_label, dtype=int)
            target_labels.append(target_label)

        # target_labels = []
        # for i in range(num_test):
        #     # lable_str = test_labels_str[i]
        #     # candidate_labels_str = np.delete(candidate_labels, np.where(candidate_labels==lable_str))
        #     target_label_str = np.random.choice(candidate_labels)
        #     target_label = list(target_label_str)
        #     target_label = np.array(target_label, dtype=int)
        #     target_labels.append(target_label)

        target_labels = np.array(target_labels, dtype=np.int)
        np.savetxt(target_label_path, target_labels, fmt="%d")


    target_labels_str = [''.join(label) for label in target_labels.astype(str)]
    qB = np.zeros([num_test, t_bit], dtype=np.float32)
    query_anchor_codes = np.zeros((num_test, bit), dtype=np.float)
    perceptibility = 0
    for it, data in enumerate(test_loader):
        queries, _, index = data
        # sample_image(queries, '{}_benign'.format(it))

        n = index[-1].item() + 1
        print(n)
        queries = queries.cuda()
        batch_size_ = index.size(0)

        anchor_codes = torch.zeros((batch_size_, bit), dtype=torch.float)
        for i in range(batch_size_):
            target_label_str = target_labels_str[index[0] + i]
            anchor_indexes = np.where(database_labels_str == target_label_str)
            anchor_indexes = np.random.choice(anchor_indexes[0], size=n_t)

            anchor_code = hash_anchor_code(
                torch.from_numpy(database_hash[anchor_indexes]))
            anchor_code = anchor_code.view(1, bit)
            anchor_codes[i, :] = anchor_code

        query_anchor_codes[it*batch_size:it*batch_size+batch_size_] = anchor_codes.numpy()

        query_adv = target_hash_adv(model, queries, anchor_codes.cuda(), epsilon, iteration=iteration)
        # queries = queries.detach().cpu().numpy()[0] * 255
        # queries = queries.astype(np.uint8)
        # queries = np.transpose(queries, (1,2,0))
        # queries = Image.fromarray(queries)
        # queries.save('0.jpg', quality=100)
        # queries = Image.open('0.jpg')
        # queries = np.array(queries, dtype=np.int)
        # query_adv = query_adv.detach().cpu().numpy()[0] * 255
        # query_adv = query_adv.astype(np.uint8)
        # query_adv = np.transpose(query_adv, (1,2,0))
        # query_adv = Image.fromarray(query_adv)
        # query_adv.save('1.jpg', quality=100)
        # query_adv = Image.open('1.jpg')
        # query_adv = np.array(query_adv, dtype=np.int)
        # print(np.max(query_adv-queries))
        # exit(0)
        # sample_image(query_adv, '{}_adv'.format(it))
        u_ind = np.linspace(it * batch_size, np.min((num_test, (it + 1) * batch_size)) - 1, batch_size_, dtype=int)

        perceptibility += F.mse_loss(queries, query_adv).data * batch_size_

        # if it > 3:
        #     end=time.time()
        #     print('Running time: %s Seconds'%(end-start))
        #     print(torch.sqrt(perceptibility/(n)))
        #     exit(0)

        if transfer:
            query_code = generate_hash(t_model, query_adv, batch_size_, t_bit)
        else:
            query_code = generate_hash(model, query_adv, batch_size_, bit)
        qB[u_ind, :] = query_code


    # qB = np.loadtxt(test_code_path, dtype=np.float)


    np.savetxt(test_code_path, qB, fmt="%d")
    # print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
    # a_map = CalcMap(query_anchor_codes, t_database_hash, target_labels, database_labels_int)
    # print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % a_map)
    # t_map = CalcMap(qB, t_database_hash, target_labels, database_labels_int)
    # print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % t_map)
    map = CalcMap(qB, t_database_hash, test_labels_int, database_labels_int)
    print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)

