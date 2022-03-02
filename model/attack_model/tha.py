import torch.nn.functional as F
import torch.nn as nn
from utils.data_provider import *
from utils.hamming_matching import *
from model.util import load_model, get_database_code, generate_code, get_alpha
from utils.data_provider import get_classes_num
from utils.util import Logger


class PrototypeNet(nn.Module):
    def __init__(self, bit, num_classes):
        super(PrototypeNet, self).__init__()

        self.feature = nn.Sequential(nn.Linear(num_classes, 4096),
                                     nn.ReLU(True), nn.Linear(4096, 512))
        self.hashing = nn.Sequential(nn.Linear(512, bit), nn.Tanh())

    def forward(self, label):
        f = self.feature(label)
        h = self.hashing(f)
        return h


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp, sn):
        # ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        # an = torch.clamp_min(sn.detach() + self.m, min=0.)

        # delta_p = 1 - self.m
        # delta_n = self.m

        # logit_p = - ap * (sp - delta_p) * self.gamma
        # logit_n = an * (sn - delta_n) * self.gamma

        ap = torch.clamp_min(- sp.detach() + 2, min=0.)
        an = torch.clamp_min(sn.detach() + 2, min=0.)

        logit_p = - ap * sp * self.gamma
        logit_n = an * sn * self.gamma
        # logit_p = - sp * self.gamma
        # logit_n = sn * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        # loss = torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)

        return loss


def select_target_label(data_labels, target_labels):
    select_index = None
    candidate_index = np.array(range(target_labels.size(0)))
    for label in data_labels:
        label_index = []
        for i in range(len(target_labels)):
            if torch.sum(label * target_labels[i]) > 0 or torch.all(target_labels[i] == 0):
                label_index.append(i)
        index = np.random.choice(np.delete(candidate_index, np.array(label_index)), size=1)
        select_index = index if select_index is None else np.concatenate((select_index, index))
    return target_labels.index_select(0, torch.from_numpy(select_index))


def similarity(batch_feature, features, batch_label, labels, bit):
    similarity_matrix = batch_feature @ features.transpose(1, 0)
    similarity_matrix = similarity_matrix / bit
    label_matrix = (batch_label.mm(labels.t()) > 0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


def target_adv_loss(noisy_output, target_hash):
    loss = -torch.mean(noisy_output * target_hash)
    # loss = noisy_output * target_hash
    # loss = (loss -2)*loss
    # loss = torch.mean(loss)
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
        loss = target_adv_loss(noisy_output, target_hash.detach())
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


def sample_image(image, name, sample_dir='sample/attack'):
    image = image.cpu().detach()[2]
    image = transforms.ToPILImage()(image.float())
    image.save(os.path.join(sample_dir, name + '.png'), quality=100)


def tha(args, epsilon=8 / 255., lr=1e-4):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    method = 'THA'

    # load model
    attack_model = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
    attack_model = attack_model if not args.adv else 'cat_{}'.format(attack_model)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    num_classes = get_classes_num(args.dataset)
    database_loader, num_database = get_data_loader(args.data_dir, args.dataset, 'database',
                                                    args.batch_size, shuffle=False)
    train_loader, num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                              args.batch_size, shuffle=True)
    test_loader, num_test = get_data_loader(args.data_dir, args.dataset, 'test',
                                            args.batch_size, shuffle=False)

    test_labels = get_data_label(args.data_dir, args.dataset, 'test')
    database_labels = get_data_label(args.data_dir, args.dataset, 'database')

    database_labels = torch.from_numpy(database_labels).float()
    target_labels = database_labels.unique(dim=0)

    database_hash, _ = get_database_code(model, database_loader, attack_model)

    if not args.adv:
        pnet_path = 'checkpoint/PrototypeNet_{}.pth'.format(attack_model)
    else:
        pnet_path = 'checkpoint/cat_PrototypeNet_{}.pth'.format(attack_model)

    if os.path.exists(pnet_path):
        pnet = load_model(pnet_path)
    else:
        print("Training PrototypeNet")
        pnet = PrototypeNet(args.bit, num_classes).cuda()
        optimizer_l = torch.optim.Adam(pnet.parameters(), lr=lr, betas=(0.5, 0.999))
        epochs = 100
        steps = 300
        # batch_size = 64
        lr_steps = epochs * steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_l,
                                                         milestones=(lr_steps / 2, lr_steps * 3 / 4), gamma=0.1)
        # criterion_l2 = torch.nn.MSELoss()
        circle_loss = CircleLoss(m=0, gamma=1)

        # hash codes of training set
        train_hash, train_labels = generate_code(model, train_loader)
        train_hash = torch.from_numpy(train_hash).cuda()
        train_labels = torch.from_numpy(train_labels).cuda()

        for epoch in range(epochs):
            for i in range(steps):
                select_index = np.random.choice(range(target_labels.size(0)), size=args.batch_size)
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()

                optimizer_l.zero_grad()
                target_hash_l = pnet(batch_target_label)
                sp, sn = similarity(target_hash_l, train_hash, batch_target_label, train_labels, args.bit)
                logloss = circle_loss(sp, sn) / args.batch_size
                regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * args.batch_size)
                loss = logloss + regterm

                loss.backward()
                optimizer_l.step()
                if i % 30 == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, logloss:{:.5f}, regterm: {:.5f}'.
                          format(epoch, i, scheduler.get_last_lr()[0], logloss, regterm))
                scheduler.step()

        torch.save(pnet, pnet_path)
        pnet.eval()

    target_label_path = 'log/target_label_tadh_{}.txt'.format(args.dataset)
    if os.path.exists(target_label_path):
        targeted_labels = np.loadtxt(target_label_path, dtype=np.int)
    else:
        print("Generating target labels")
        targeted_labels = np.zeros([num_test, num_classes])
        for data in test_loader:
            _, label, index = data
            batch_target_label = select_target_label(label, target_labels)
            targeted_labels[index.numpy(), :] = batch_target_label.numpy()
        np.savetxt(target_label_path, targeted_labels, fmt="%d")

    qB = np.zeros([num_test, args.bit], dtype=np.float32)
    qB_ori = np.zeros([num_test, args.bit], dtype=np.float32)
    query_prototype_codes = np.zeros((num_test, args.bit), dtype=np.float)
    perceptibility = 0
    for it, data in enumerate(test_loader):
        queries, _, index = data

        n = index[-1].item() + 1
        print(n)
        queries = queries.cuda()
        batch_size_ = index.size(0)

        batch_target_label = targeted_labels[index.numpy(), :]
        batch_target_label = torch.from_numpy(batch_target_label).float().cuda()

        batch_prototype_codes = pnet(batch_target_label)
        prototype_codes = torch.sign(batch_prototype_codes)
        query_prototype_codes[index.numpy(), :] = prototype_codes.cpu().data.numpy()
        query_adv = target_hash_adv(model, queries, prototype_codes, epsilon, iteration=args.iteration)

        perceptibility += F.mse_loss(queries, query_adv).data * batch_size_
        query_code = model(query_adv)
        query_code = torch.sign(query_code)
        qB[index.numpy(), :] = query_code.cpu().data.numpy()
        qB_ori[index.numpy(), :] = model(queries).sign().cpu().data.numpy()

        # sample_image(queries, '{}_benign'.format(it))
        # sample_image(query_adv, '{}_adv'.format(it))

    # save code
    np.save(os.path.join('log', attack_model, '{}_code.npy'.format(method)), qB)

    logger = Logger(os.path.join('log', attack_model), '{}.txt'.format(method))
    logger.log('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))

    # map_ = cal_map(database_hash, qB_ori, database_labels.numpy(), test_labels, 5000)
    # logger.log('Ori MAP(retrieval database): {}'.format(map_))
    map_ = cal_map(database_hash, qB, database_labels.numpy(), test_labels, 5000)
    logger.log('THA MAP(retrieval database): {}'.format(map_))
    # p_map = cal_map(database_hash, query_prototype_codes, database_labels.numpy(), test_labels, 5000)
    # logger.log('Theory MAP(retrieval database): {}'.format(p_map))
    # t_map = cal_map(database_hash, qB, database_labels.numpy(), targeted_labels, 5000)
    # logger.log('THA t-MAP(retrieval database): {}'.format(t_map))
    # t_map = cal_map(database_hash, query_prototype_codes, database_labels.numpy(), targeted_labels, 5000)
    # logger.log('Theory t-MAP(retrieval database): {}'.format(t_map))
