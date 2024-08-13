import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm




class lfw_dataset(Dataset):
    def __init__(self, img_dir, txt_dir, img_size):
        self.img_size = img_size
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
        img1 = cv2.imread(dir1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(dir2, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (self.img_size, self.img_size))
        img2 = cv2.resize(img2, (self.img_size, self.img_size))

        return img1, img2, label


class lfw_evaluator():
    def __init__(self, img_dir, txt_dir, img_size):
        lfw_set = lfw_dataset(img_dir, txt_dir, img_size)
        self.img_size = img_size
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
        score_list = []
        label_list = []
        with torch.no_grad():
            for i, (img1, img2, pos) in tqdm(enumerate(self.lfw_loader)):
                B = pos.size(0)
                feat1 = encoder(img1)  # [B,512]
                feat2 = encoder(img2)
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



class HOG_descriptor():
    def __init__(self, cell_size=16, bin_size=8):
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        #assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self, img, visualize=False):
        height, width = img.shape
        img = 255 * np.sqrt(img / float(np.max(img)))
        gradient_magnitude, gradient_angle = self.global_gradient(img)
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        hog_vector = np.array(hog_vector, dtype=np.float32).reshape(-1)
        if visualize:
            hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
            return hog_vector, hog_image
        else:
            return hog_vector

    def global_gradient(self, img):
        gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


class HOG_wrapper:
    def __init__(self, hog):
        self.hog = hog

    # mimic .to() function of nn.Module
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        pass

    def __call__(self, img):
        if img.min() < 0 and img.min() >= -1:
            img = 0.5 * (img + 1)  # normalize to 0~1
        if len(img.shape) == 4:    # B, C, H, W
            img = img.mean(dim=1)  # B, H, W
        img = img.cpu().numpy()
        B, H, W = img.shape

        vectors = []
        for i in range(B):
            vectors.append(self.hog.extract(img[i], visualize=False))
        vectors = np.vstack(vectors)
        vectors = torch.from_numpy(vectors)
        if hasattr(self, "device"):
            vectors = vectors.to(self.device)
        return vectors


if __name__ == "__main__":
    device = torch.device('cuda:0')
    hog = HOG_descriptor(cell_size=16, bin_size=16)
    encoder = HOG_wrapper(hog)
    # encoder = encoder.to(device)
    # img_dir = '/home/yoon/datasets/face/FaceXZoo/lfw/lfw_crop'  # directory for lfw images
    # txt_dir = '/home/yoon/datasets/face/FaceXZoo/lfw/pairs.txt'  # directory for lfw pairs.txt
    # evaluator = lfw_evaluator(img_dir, txt_dir, 160)
    # acc, std = evaluator.evaluate(encoder)
    # print(f"acc:{acc * 100}%, std:{std * 100}%")
    imdir = "/data/New_Projects/ZOIA/results/lfw/FaceNet/BB_top10_k4_iters200/attack_images/Aaron_Peirsol_0001.jpg"
    img = cv2.imread(imdir, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (192, 192))
    vector, image = hog.extract(img, visualize=True)
    print(vector.shape)
    fig,ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].imshow(Image.open(imdir))
    ax[1].imshow(image)
    ax[0].set_title("Original Image", fontsize=17)
    ax[1].set_title("HOG Feature", fontsize=17)
    plt.show()
