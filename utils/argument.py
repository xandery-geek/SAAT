
def add_base_arguments(parser):
    # arguments for base config
    parser.add_argument('--device', dest='device', type=str, default='0', help='gpu device')
    return parser


def add_dataset_arguments(parser):
    
    # arguments for dataset
    parser.add_argument('--data_dir', dest='data_dir', default='../data/', help='path of the dataset')
    parser.add_argument('--dataset', dest='dataset', default='NUS-WIDE',
                        choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],
                        help='name of the dataset')
    return parser


def add_model_arguments(parser):
    # arguments for hashing model
    parser.add_argument('--hash_method', dest='hash_method', default='DPH', 
                        choices=['DPH', 'DPSH', 'CSQ', 'HashNet', 'DPN', 'HSWD', 'Ortho'],
                        help='deep hashing methods')
    parser.add_argument('--backbone', dest='backbone', default='AlexNet',
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50'],
                        help='backbone network')
    parser.add_argument('--bit', dest='bit', type=int, default=32, help='length of the hashing code')
    return parser


def add_attack_arguments(parser):
    # arguments for attack
    parser.add_argument('--attack_method', dest='attack_method', default='mainstay', help='name of attack method')
    parser.add_argument('--targeted', dest='targeted', action="store_true", default=False, help='targeted attack')
    parser.add_argument('--iteration', dest='iteration', type=int, default=100, help='number of training iteration')
    parser.add_argument('--retrieve', dest='retrieve', action="store_true", default=False, help='retrieve images')
    parser.add_argument('--sample', dest='sample', action="store_true", default=False, help='sample adversarial examples')
    return parser


def add_defense_arguments(parser):
    parser.add_argument('--adv_method', dest='adv_method', type=str, default='saat', choices=['saat', 'atrdh'],
                        help='name of adversarial training method')
    parser.add_argument('--lambda', dest='p_lambda', type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument('--mu', dest='p_mu', type=float, default=1e-4, help='mu for quantization loss')
    return parser