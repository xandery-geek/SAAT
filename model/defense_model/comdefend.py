import os
import argparse
import shutil
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from utils.util import check_dir
from utils.data_provider import get_data_loader


class Encoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 12, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.features(x)


class Decoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Decoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.features(x)


class ComDefend(object):
    def __init__(self, args, ckpt=None):
        super(ComDefend, self).__init__()

        self.args = args
        self.encoder = Encoder()
        self.decoder = Decoder()

        if ckpt:
            print("Loading ckpt from {}".format(ckpt))
            ckpt = torch.load(ckpt)
            self.args = ckpt['args']
            self.optimizer = ckpt['optimizer']
            self.encoder.load_state_dict(ckpt['encoder'])
            self.decoder.load_state_dict(ckpt['decoder'])
        else:
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            self.optimizer = optim.Adam(params, lr=self.args.lr, betas=(0.5, 0.999))

    def _train_epoch(self, trainloader):
        self.encoder.train()
        self.decoder.train()

        total_loss = 0
        count = 0
        flag = False
        for x, _, _ in trainloader:
            x = x.cuda()

            linear_code = self.encoder(x)
            noisy_code = linear_code - torch.randn(linear_code.size()).cuda() * self.args.std
            binary_code = torch.sigmoid(noisy_code)
            recons_x = self.decoder(binary_code)
            loss = ((recons_x - x) ** 2).mean() + (binary_code ** 2).mean() * 0.0001

            if not flag:
                flag = True
                img_tensor = recons_x
                ori_img_tensor = x

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if count % self.args.interval == 0:
                print("training loss: {:.6f}".format(loss.item()))

            count += 1
        print("training average loss: {:.6f}".format(total_loss / float(count)))

        return img_tensor, ori_img_tensor

    def _test_epoch(self, testloader):
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0
        count = 0
        flag = False
        img_tensor = False
        ori_img_tensor = False
        for x, _, _ in testloader:
            x = x.cuda()

            linear_code = self.encoder(x)
            noisy_code = linear_code - torch.randn(linear_code.size()).cuda() * args.std
            binary_code = torch.round(torch.sigmoid(noisy_code))
            recons_x = self.decoder(binary_code)

            if not flag:
                flag = True
                img_tensor = recons_x
                ori_img_tensor = x

            loss = ((recons_x - x) ** 2).mean()
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / float(count)

        print("test average loss: {:.6f}".format(avg_loss))
        return avg_loss, img_tensor, ori_img_tensor
    
    def save_checkpoint(self, is_best, filepath):
        check_dir(filepath, isdir=False)

        state_dict = {"encoder": self.encoder.state_dict(),
                        "decoder": self.decoder.state_dict(),
                        "optimizer": self.optimizer, 
                        "args": self.args}
        torch.save(state_dict, filepath)

        if is_best:
            shutil.copyfile(filepath, '/'.join(filepath.split('/')[:-1]) + '/best.pth')

    @staticmethod
    def save_image(save_path, tensor, ori_tensor):
        check_dir(save_path, isdir=False)

        img = tensor.data.cpu().numpy()
        img = img.transpose(0, 2, 3, 1) * 255.0
        img = np.array(img).astype(np.uint8)
        img = np.concatenate(img, 1)

        ori_img = ori_tensor.data.cpu().numpy()
        ori_img = ori_img.transpose(0, 2, 3, 1) * 255.0
        ori_img = np.array(ori_img).astype(np.uint8)
        ori_img = np.concatenate(ori_img, 1)

        vis = np.concatenate(np.array([ori_img, img]), 0)
        img_pil = Image.fromarray(vis)
        img_pil.save(save_path)

    def train(self, trainloader, testloader):
        best_loss = 1e10
        isbest = False

        for epoch in range(self.args.max_epochs):
            print("=============== Epoch: {} ===============".format(epoch))
            train_img, train_ori = self._train_epoch(trainloader)
            test_loss, test_img, test_ori = self._test_epoch(testloader)
            
            img_path = "checkpoint/comdefend/{}/train_{}.jpg".format(self.args.dataset, epoch)
            self.save_image(img_path, train_img[:8], train_ori[:8])
            img_path = "checkpoint/comdefend/{}/test_{}.jpg".format(self.args.dataset, epoch)
            self.save_image(img_path, test_img[:8], test_ori[:8])

            if best_loss > test_loss:
                print("best loss update, pre: {:.6f}, cur: {:.6f}".format(best_loss, test_loss))
                best_loss = test_loss
                isbest = True
            else:
                isbest = False

            filepath = "checkpoint/comdefend/{}/comdefend_{}.pth".format(self.args.dataset, epoch)
            self.save_checkpoint(isbest, filepath)
    
    def cuda(self):
        self.encoder.cuda()
        self.decoder.cuda()
    
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def apply(self, x):
        linear_code = self.encoder(x)
        noisy_code = linear_code - torch.randn(linear_code.size()).cuda() * self.args.std
        binary_code = torch.sigmoid(noisy_code)
        recons_x = self.decoder(binary_code)
        return recons_x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch comdefend Training')

    parser.add_argument('--data_dir', dest='data_dir', default='../data/', type=str)
    parser.add_argument('--dataset', default='NUS-WIDE', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--interval', default=100, type=int)
    parser.add_argument('--std', default=20.0, type=float)
    parser.add_argument('--device', default="0", type=str)
    parser.add_argument('--seed', default=100, type=int)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model = ComDefend(args)
    model.cuda()
    
    train_loader, _ = get_data_loader(args.data_dir, args.dataset, 'train',
                                        args.batch_size, shuffle=True)
    test_loader, _ = get_data_loader(args.data_dir, args.dataset, 'test',
                                        args.batch_size//2, shuffle=False)
    
    model.train(train_loader, test_loader)
