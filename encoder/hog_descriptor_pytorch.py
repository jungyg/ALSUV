import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class lfw_dataset(Dataset):
    def __init__(self, img_dir, txt_dir, img_size, trf, flipcat):
        self.img_size = img_size
        self.trf = trf
        self.flipcat = flipcat
        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.loader = []
        f = open(txt_dir, 'r')
        for i, line in enumerate(f.readlines()):
            if i == 0: continue
            elem = line[:-1].split('\t')
            if len(elem) == 3:
                dir1 = os.path.join(img_dir, elem[0], elem[0] + '_{:04d}.jpg'.format(int(elem[1])))
                dir2 = os.path.join(img_dir, elem[0], elem[0] + '_{:04d}.jpg'.format(int(elem[2])))
                self.loader.append((dir1, dir2, True))
            elif len(elem) == 4:
                dir1 = os.path.join(img_dir, elem[0], elem[0] + '_{:04d}.jpg'.format(int(elem[1])))
                dir2 = os.path.join(img_dir, elem[2], elem[2] + '_{:04d}.jpg'.format(int(elem[3])))
                self.loader.append((dir1, dir2, False))

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        dir1, dir2, label = self.loader[idx]
        img1 = Image.open(dir1).resize((self.img_size, self.img_size))
        img2 = Image.open(dir2).resize((self.img_size, self.img_size))
        img1 = self.trf(img1)
        img2 = self.trf(img2)

        return img1, img2, label


class lfw_evaluator():
    def __init__(self, img_dir, txt_dir, img_size, device, flipcat=False):
        trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.FloatTensor([0.5,0.5,0.5]), torch.FloatTensor([0.5,0.5,0.5])),
        ])
        lfw_set = lfw_dataset(img_dir, txt_dir, img_size, trf, flipcat)
        self.img_size = img_size
        self.device = device
        self.flipcat = flipcat
        self.lfw_loader = DataLoader(lfw_set, batch_size=60, shuffle=False, num_workers=4)

    def getThreshold(self, score_list, label_list, num_thresholds=1000):
        """Get the best threshold by train_score_list and train_label_list.
        Args:
            score_list(ndarray): the score list of all pairs.
            label_list(ndarray): the label list of all pairs.
            num_thresholds(int): the number of threshold that used to compute roc.
        Returns:
            best_thres(float): the best threshold that computed by train set.
        """
        pos_score_list = score_list[label_list == 1]
        neg_score_list = score_list[label_list == 0]
        pos_pair_nums = pos_score_list.size
        neg_pair_nums = neg_score_list.size
        score_max = np.max(score_list)
        score_min = np.min(score_list)
        score_span = score_max - score_min
        step = score_span / num_thresholds
        threshold_list = score_min + step * np.array(range(1, num_thresholds + 1))
        fpr_list = []
        tpr_list = []
        for threshold in threshold_list:
            fpr = np.sum(neg_score_list > threshold) / neg_pair_nums
            tpr = np.sum(pos_score_list > threshold) / pos_pair_nums
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        best_index = np.argmax(tpr - fpr)
        best_thres = threshold_list[best_index]
        return best_thres

    def evaluate(self, encoder, num_threshold=1000):
        encoder.eval()
        score_list = []
        label_list = []
        with torch.no_grad():
            for i, (img1, img2, pos) in enumerate(self.lfw_loader):
                B = pos.size(0)
                img1 = img1.to(self.device)  # [B,3,112,112] if not flipcat
                img2 = img2.to(self.device)  # [B,2,3,112,112] if flipcat
                feat1 = encoder(img1).reshape(B, -1)
                feat2 = encoder(img2).reshape(B, -1)
                feat1 = F.normalize(feat1, dim=1)
                feat2 = F.normalize(feat2, dim=1)
                for b in range(B):
                    sim = torch.dot(feat1[b], feat2[b])  # scalar value
                    score_list.append(sim.item())
                    label_list.append(pos[b].item())
        score_list = np.array(score_list).reshape(10, 600)
        label_list = np.array(label_list).reshape(10, 600)

        subset_train = np.array([True] * 10)
        accu_list = []
        for subset_idx in range(10):
            test_score_list = score_list[subset_idx]
            test_label_list = label_list[subset_idx]
            subset_train[subset_idx] = False
            train_score_list = score_list[subset_train].flatten()
            train_label_list = label_list[subset_train].flatten()
            subset_train[subset_idx] = True
            best_thres = self.getThreshold(train_score_list, train_label_list)
            positive_score_list = test_score_list[test_label_list == 1]
            negtive_score_list = test_score_list[test_label_list == 0]
            true_pos_pairs = np.sum(positive_score_list > best_thres)
            true_neg_pairs = np.sum(negtive_score_list < best_thres)
            accu_list.append((true_pos_pairs + true_neg_pairs) / 600)
        mean = np.mean(accu_list)
        std = np.std(accu_list, ddof=1) / np.sqrt(10)  # ddof=1, division 9.

        return mean, std


class HOGLayer(nn.Module):
    def __init__(self, nbins=10, pool=8, max_angle=math.pi, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.max_angle = max_angle
        mat = torch.FloatTensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        with torch.no_grad():
            gxy = F.conv2d(x, self.weight, None, self.stride,
                            self.padding, self.dilation, 1)
            #2. Mag/ Phase
            mag = gxy.norm(dim=1)
            norm = mag[:,None,:,:]
            phase = torch.atan2(gxy[:,0,:,:], gxy[:,1,:,:])

            #3. Binning Mag with linear interpolation
            phase_int = phase / self.max_angle * self.nbins
            phase_int = phase_int[:,None,:,:]

            n, c, h, w = gxy.shape
            out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
            out.scatter_(1, phase_int.floor().long()%self.nbins, norm)
            out.scatter_add_(1, phase_int.ceil().long()%self.nbins, 1 - norm)

            return self.pooler(out)




class HOGLayerMoreComplicated(nn.Module):
    def __init__(self, nbins=10, pool=8, max_angle=math.pi, stride=1, padding=1, dilation=1,
                 mean_in=False, max_out=False):
        super(HOGLayerMoreComplicated, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.max_angle = max_angle
        self.max_out = max_out
        self.mean_in = mean_in

        mat = torch.FloatTensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        if self.mean_in:
            return self.forward_v1(x)
        else:
            return self.forward_v2(x)

    def forward_v1(self, x):
        if x.size(1) > 1:
            x = x.mean(dim=1)[:,None,:,:]

        gxy = F.conv2d(x, self.weight, None, self.stride,
                       self.padding, self.dilation, 1)
        # 2. Mag/ Phase
        mag = gxy.norm(dim=1)
        norm = mag[:, None, :, :]
        phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])

        # 3. Binning Mag with linear interpolation
        phase_int = phase / self.max_angle * self.nbins
        phase_int = phase_int[:, None, :, :]

        n, c, h, w = gxy.shape
        out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
        out.scatter_(1, phase_int.floor().long() % self.nbins, norm)
        out.scatter_add_(1, phase_int.ceil().long() % self.nbins, 1 - norm)

        return self.pooler(out)

    def forward_v2(self, x):
        batch_size, in_channels, height, width = x.shape
        weight = self.weight.repeat(3, 1, 1, 1)
        gxy = F.conv2d(x, weight, None, self.stride,
                        self.padding, self.dilation, groups=in_channels)

        gxy = gxy.view(batch_size, in_channels, 2, height, width)

        if self.max_out:
            gxy = gxy.max(dim=1)[0][:,None,:,:,:]

        #2. Mag/ Phase
        mags = gxy.norm(dim=2)
        norms = mags[:,:,None,:,:]
        phases = torch.atan2(gxy[:,:,0,:,:], gxy[:,:,1,:,:])

        #3. Binning Mag with linear interpolation
        phases_int = phases / self.max_angle * self.nbins
        phases_int = phases_int[:,:,None,:,:]

        out = torch.zeros((batch_size, in_channels, self.nbins, height, width),
                          dtype=torch.float, device=gxy.device)
        out.scatter_(2, phases_int.floor().long()%self.nbins, norms)
        out.scatter_add_(2, phases_int.ceil().long()%self.nbins, 1 - norms)

        out = out.view(batch_size, in_channels * self.nbins, height, width)
        out = torch.cat((self.pooler(out), self.pooler(x)), dim=1)

        return out


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    hog = HOGLayerMoreComplicated(nbins=4, pool=8)
    encoder = hog.to(device)

