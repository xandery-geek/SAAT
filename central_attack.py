import torch.nn.functional as F

from utils.data_provider import *
from utils.hamming_matching import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def adv_loss(noisy_output, target_hash):
    loss = torch.mean(noisy_output * target_hash)
    # sim = noisy_output * target_hash
    # w = (sim>-0.5).int()
    # sim = w*(sim+2)*sim
    # loss = torch.mean(sim)
    return loss


def hash_adv(model, query, target_hash, epsilon, step=1, iteration=2000, randomize=False):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        noisy_output = model(query + delta)
        loss = adv_loss(noisy_output, target_hash.detach())
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


def hash_center_code(y, B, L, bit):
    code = torch.zeros(y.size(0), bit).cuda()
    for i in range(y.size(0)):
        l = y[i].repeat(L.size(0), 1)
        w = torch.sum(l * L, dim=1) / torch.sum(torch.sign(l + L), dim=1)
        w1 = w.repeat(32, 1).t()
        w2 = 1 - torch.sign(w1)
        code[i] = torch.sign(torch.sum(w1 * B - w2 * B, dim=0))
    return code


def sample_image(image, name, sample_dir='sample/attack'):
    image = image.cpu().detach()[2]
    image = transforms.ToPILImage()(image.float())
    image.save(os.path.join(sample_dir, name + '.png'), quality=100)


classes_dic = {'CIFAR-10': 10, 'ImageNet': 100, 'FLICKR-25K': 38, 'NUS-WIDE': 21, 'MS-COCO': 80}
dataset = 'NUS-WIDE'
data_dir = '../data/{}'.format(dataset)
database_file = 'database_img.txt'
train_file = 'train_img.txt'
test_file = 'test_img.txt'
database_label = 'database_label.txt'
train_label = 'train_label.txt'
test_label = 'test_label.txt'
num_classes = classes_dic[dataset]
model_name = 'DPH'
backbone = 'AlexNet'
defense = ''
batch_size = 32
bit = 32
epsilon = 8 / 255.0
iteration = 100

lr = 1e-4
transfer = False

dset_database = HashingDataset(data_dir, database_file, database_label)
dset_train = HashingDataset(data_dir, train_file, train_label)
dset_test = HashingDataset(data_dir, test_file, test_label)
database_loader = DataLoader(dset_database, batch_size=batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=4)
num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

database_labels = load_label(database_label, data_dir)
train_labels = load_label(train_label, data_dir)
test_labels = load_label(test_label, data_dir)
target_labels = database_labels.unique(dim=0)

model_path = 'checkpoint/{}{}_{}_{}_{}.pth'.format(defense, dataset, model_name, backbone, bit)
model = load_model(model_path)
database_code_path = 'log/{}database_code_{}_{}_{}_{}.txt'.format(defense, dataset, model_name, backbone, bit)

if transfer:
    t_model_name = 'DPH'
    t_bit = 32
    t_backbone = 'ResNet18'
    t_model_path = 'checkpoint/{}{}_{}_{}_{}_{}.pth'.format(defense, dataset, t_model_name, t_backbone, t_bit)
    t_model = load_model(t_model_path)
else:
    t_model_name = model_name
    t_bit = bit
    t_backbone = backbone
t_database_code_path = 'log/{}database_code_{}_{}_{}_{}.txt'.format(defense, dataset, t_model_name, t_backbone, t_bit)
target_label_path = 'log/target_label_attack_{}.txt'.format(dataset)
test_code_path = 'log/test_code_{}_attack_{}.txt'.format(dataset, t_bit)

if os.path.exists(database_code_path):
    database_hash = np.loadtxt(database_code_path, dtype=np.float)
else:
    database_hash = generate_hash_code(model, database_loader, num_database, bit)
    np.savetxt(database_code_path, database_hash, fmt="%d")
if os.path.exists(t_database_code_path):
    t_database_hash = np.loadtxt(t_database_code_path, dtype=np.float)
else:
    t_database_hash = generate_hash_code(t_model, database_loader, num_database, t_bit)
    np.savetxt(t_database_code_path, t_database_hash, fmt="%d")
print('database hash codes prepared!')

train_B, train_L = generate_code_label(model, train_loader, num_train, bit, num_classes)
qB = np.zeros([num_test, t_bit], dtype=np.float32)
perceptibility = 0
for it, data in enumerate(test_loader):
    queries, labels, index = data
    queries = queries.cuda()
    labels = labels.cuda()
    batch_size_ = index.size(0)

    n = index[-1].item() + 1
    print(n)

    center_codes = hash_center_code(labels, train_B, train_L, bit)
    query_adv = hash_adv(model, queries, center_codes, epsilon, iteration=iteration)

    perceptibility += F.mse_loss(queries, query_adv).data * batch_size_

    if transfer:
        query_code = t_model(query_adv)
    else:
        query_code = model(query_adv)
    query_code = torch.sign(query_code)
    qB[index.numpy(), :] = query_code.cpu().data.numpy()

    # sample_image(queries, '{}_benign'.format(it))
    # sample_image(query_adv, '{}_adv'.format(it))

# np.savetxt(test_code_path, qB, fmt="%d")
# print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
# p_map = CalcMap(query_prototype_codes, t_database_hash, targeted_labels, database_labels.numpy())
# print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % p_map)
# t_map = CalcMap(qB, t_database_hash, targeted_labels, database_labels.numpy())
# print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % t_map)
map = CalcTopMap(qB, database_hash, test_labels.numpy(), database_labels.numpy(), 5000)
print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
map = CalcMap(qB, database_hash, test_labels.numpy(), database_labels.numpy())
print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
