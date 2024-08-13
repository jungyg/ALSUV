import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse
from encoder.fetch import fetch_encoder

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
        if self.flipcat:
            img1 = torch.cat((img1, self.flip(img1)), dim=0)  # [2,3,112,112]
            img2 = torch.cat((img2, self.flip(img2)), dim=0)
        return img1, img2, label


class lfw_evaluator():
    def __init__(self, img_dir, txt_dir, img_size, device, flipcat=False):
        trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
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
            for i, (img1, img2, label) in enumerate(self.lfw_loader):
                B = label.size(0)
                img1 = img1.to(self.device)  # [B,3,112,112] if not flipcat
                img2 = img2.to(self.device)  # [B,2,3,112,112] if flipcat
                if not self.flipcat:
                    feat1 = encoder(img1)  # [B,512]
                    feat2 = encoder(img2)
                if self.flipcat:
                    img1 = img1.view(2 * B, 3, self.img_size, self.img_size)
                    img2 = img2.view(2 * B, 3, self.img_size, self.img_size)
                    feat1 = encoder(img1)  # [2B,512]
                    feat2 = encoder(img2)
                    feat1 = feat1.view(B, -1)  # [2B,512] >> [B,1024]
                    feat2 = feat2.view(B, -1)
                feat1 = F.normalize(feat1, dim=1)
                feat2 = F.normalize(feat2, dim=1)
                for b in range(B):
                    sim = torch.dot(feat1[b], feat2[b])  # scalar value
                    score_list.append(sim.item())
                    label_list.append(label[b].item())
        score_list = np.array(score_list).reshape(10, 600)
        label_list = np.array(label_list).reshape(10, 600)

        subset_train = np.array([True] * 10)
        accu_list = []
        threshold_list = []
        for subset_idx in range(10):
            test_score_list = score_list[subset_idx]
            test_label_list = label_list[subset_idx]
            subset_train[subset_idx] = False
            train_score_list = score_list[subset_train].flatten()
            train_label_list = label_list[subset_train].flatten()
            subset_train[subset_idx] = True
            best_thres = self.getThreshold(train_score_list, train_label_list)
            threshold_list.append(best_thres)
            positive_score_list = test_score_list[test_label_list == 1]
            negtive_score_list = test_score_list[test_label_list == 0]
            true_pos_pairs = np.sum(positive_score_list > best_thres)
            true_neg_pairs = np.sum(negtive_score_list < best_thres)
            accu_list.append((true_pos_pairs + true_neg_pairs) / 600)
        mean = np.mean(accu_list)
        std = np.std(accu_list, ddof=1) / np.sqrt(10)  # ddof=1, division 9.
        print(np.mean(np.array(threshold_list)))

        return mean, std




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--align', type=str, default='mtcnn', help='["mtcnn", "FXZoo"]')
    parser.add_argument('--target_encoder', type=str, default='VGGNet19', choices=['VGGNet19', 'SwinTransformer' 'FaceNet', 'ResNet50'])
    parser.add_argument('--device_id', type=int, default=0)

    args = parser.parse_args()
    args.device = f'cuda:{args.device_id}'
    if args.target in ['FaceNet', 'VGGNet19']:
        args.align = 'mtcnn'
    elif args.target in ['ResNet100', 'SwinTransformer']:
        args.align = 'FXZoo'

    args.img_dir = f'./datasets/face/lfw/{args.align}_aligned'  # directory for lfw images
    args.txt_dir = './datasets/face/lfw/pairs.txt'  # directory for lfw pairs.txt

    args.encoder_type = "VGGNet19"  # TODO: which encoder to evaluate?
    encoder = fetch_encoder(args.target_encoder).to(args.device_id)

    img_dir = './datasets/face/FaceXZoo/lfw/lfw_crop'  # directory for lfw images
    txt_dir = './datasets/face/FaceXZoo/lfw/pairs.txt'  # directory for lfw pairs.txt
    evaluator = lfw_evaluator(args.img_dir, args.txt_dir, encoder.img_size, args.device_id, flipcat=False)
    acc, std = evaluator.evaluate(encoder)
    print(f"acc:{acc * 100}%, std:{std * 100}%")


