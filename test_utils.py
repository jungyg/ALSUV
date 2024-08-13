import os
import json
import random
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import yaml
import glob
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils, transforms

from dataset import dataset_parser
from generator.stylegan_utils import StyleGANWrapper
from encoder.fetch import fetch_encoder
from encoder.blackbox_encoder import WhiteboxEncoder, BlackboxEncoder
from utils import cosine_similarity, str2bool
from torch.utils.data import Dataset, DataLoader



class lfw_dataset(Dataset):
    def __init__(self, img_dirs, trf):
        self.img_dirs = img_dirs
        self.trf = trf

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        dir = self.img_dirs[idx]
        imname = dir.split('/')[-1]
        img = Image.open(dir)
        img = self.trf(img)
        return img, imname


class cfp_dataset(Dataset):
    def __init__(self, img_dirs, trf):
        self.img_dirs = img_dirs
        self.trf = trf

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        dir = self.img_dirs[idx]
        imname = dir.split('/')[-3]
        img = Image.open(dir)
        img = self.trf(img)
        return img, imname

class agedb_30_dataset(Dataset):
    def __init__(self, img_dirs, trf):
        self.img_dirs = img_dirs
        self.trf = trf

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        dir = self.img_dirs[idx]
        imname = dir.split('/')[-1]
        img = Image.open(dir)
        img = self.trf(img)
        return img, imname



class lfw_evaluator():
    def __init__(self, args):
        self.trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.device = args.device
        self.save_dir = args.save_dir

        ## encoder used for generation
        self.src_encoder_name = args.src_encoder
        self.src_encoder = fetch_encoder(self.src_encoder_name).to(args.device)
        self.src_encoder = BlackboxEncoder(self.src_encoder, img_size=self.src_encoder.img_size)
        self.src_encoder.eval()

        ## encoder to attack
        self.tgt_encoder_name = args.tgt_encoder
        self.tgt_encoder = fetch_encoder(self.tgt_encoder_name).to(args.device)
        self.tgt_encoder = BlackboxEncoder(self.tgt_encoder, img_size=self.tgt_encoder.img_size)
        self.tgt_encoder.eval()

        self.val_encoder_name = args.val_encoder
        self.val_encoder = fetch_encoder(self.val_encoder_name).to(args.device)
        self.val_encoder = BlackboxEncoder(self.val_encoder, img_size=self.val_encoder.img_size)
        self.val_encoder.eval()

        self.atk_img_dir = os.path.join(args.save_dir, 'attack_images')

        with open('./config/encoder_config.yaml') as fp:
            self.tgt_encoder_conf = yaml.load(fp, Loader=yaml.FullLoader)
            self.tgt_encoder_conf = self.tgt_encoder_conf[args.tgt_encoder]

        with open('./config/dataset_config.yaml') as fp:
            self.dataset_conf = yaml.load(fp, Loader=yaml.FullLoader)
            self.dataset_conf = self.dataset_conf[args.dataset]


    @torch.no_grad()
    def verification(self, args):
        atk_dirs, atk_imnames = self._get_top1_sample(args)
        pos_feats, pos_imnames = self._get_feats(args=args, 
                                                 encoder=self.tgt_encoder,
                                                 type='positive',
                                                 flip=self.flip)
        atk_feats, _ = self._get_feats(args=args,
                                        encoder=self.tgt_encoder,
                                        type='attack',
                                        atk_dirs=atk_dirs,
                                        atk_imnames= atk_imnames)
        atk_scores_type1, atk_scores_type2 = self._get_ver_atk_scores(pos_feats, pos_imnames, atk_feats, atk_imnames)
        self._save_verification(atk_scores_type1, atk_scores_type2)     

    @torch.no_grad()
    def identification(self, args):
        atk_dirs, atk_imnames = self._get_top1_sample(args)
        gall_feats, gall_imnames = self._get_feats(args=args, 
                                                 encoder=self.tgt_encoder,
                                                 type='gallery',
                                                 flip=self.flip)
        atk_feats, _ = self._get_feats(args=args, 
                                      encoder=self.tgt_encoder,
                                        type='attack',
                                        atk_dirs=atk_dirs,
                                        atk_imnames= atk_imnames)
        inc_type1, exc_type1 = self._get_iden_atk_scores(gall_feats, gall_imnames, atk_feats, atk_imnames)
        self._save_identification(inc_type1, exc_type1)

    @torch.no_grad()
    def _get_feats(self, args, encoder, type, atk_dirs=None, atk_imnames=None, flip=True):
        img_dir = self.dataset_conf['image_dir']+ f'/{args.align}_aligned'
        targets_txt = self.dataset_conf['targets_txt']
        with open(targets_txt, 'r') as fp:
            lines = fp.readlines()
        if type == 'positive':
            pos_ids = [l.strip()[:-9] for l in lines]
            pos_dirs = []
            for name in pos_ids:
                pos_dirs += glob.glob(os.path.join(img_dir, name, '*'))
            pos_set = lfw_dataset(pos_dirs, self.trf)
            pos_loader = DataLoader(pos_set, batch_size=60, shuffle=False, num_workers=2)
            features = torch.FloatTensor([])
            imnames = []
            for _, (img, name) in tqdm(enumerate(pos_loader)):
                img = img.to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
                imnames += list(name)
        elif type == 'attack':
            assert atk_dirs
            assert atk_imnames
            features = torch.FloatTensor([])
            imnames = []
            for dir in atk_dirs:
                img = self.trf(Image.open(dir))
                img = img.unsqueeze(0).to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
        elif type == 'gallery':
            all_ids = os.listdir(img_dir)
            gall_dirs = []
            for name in all_ids:
                gall_dirs += glob.glob(os.path.join(img_dir, name, '*'))
            gall_set = lfw_dataset(gall_dirs, self.trf)
            gall_loader = DataLoader(gall_set, batch_size=60, shuffle=False, num_workers=2)

            features = torch.FloatTensor([])
            imnames = []
            for i, (img, name) in tqdm(enumerate(gall_loader)):
                img = img.to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
                imnames += list(name)
        else:
            raise ValueError(f'{type} is not a valid type; ["positive","negative","attack"]')

        return features, imnames


    @torch.no_grad()
    def _get_ver_atk_scores(self, pos_feats, pos_imnames, atk_feats, atk_imnames):
        pos_idens = ['_'.join(n.split('_')[:-1]) for n in pos_imnames]
        atk_idens = ['_'.join(n.split('_')[:-1]) for n in atk_imnames]
        pos_idens = np.array([pos_idens])
        atk_idens = np.array([atk_idens])
        pos_imnames = np.array([pos_imnames])  # [1, 3166]
        atk_imnames = np.array([atk_imnames])  # [1, 200]
        mask_type1 = (atk_imnames.T == pos_imnames) # [200, 3166]
        mask_type2 = (atk_idens.T == pos_idens)     # [200, 3166]
        mask_type2 &= ~mask_type1                   # exclude type1 attacks

        # compute cosine similarity
        scores = cosine_similarity(atk_feats, pos_feats)

        scores_type1 = torch.FloatTensor([])
        scores_type2 = torch.FloatTensor([])
        for i in range(scores.shape[0]):
            mask1, mask2 = mask_type1[i], mask_type2[i]
            scores_type1 = torch.cat((scores_type1, scores[i][mask1]), dim=0)
            scores_type2 = torch.cat((scores_type2, scores[i][mask2]), dim=0)

        return scores_type1.numpy(), scores_type2.numpy()


    @torch.no_grad()
    def _get_iden_atk_scores(self, gall_feats, gall_imnames, atk_feats, atk_imnames):
        gall_idens = ['_'.join(n.split('_')[:-1]) for n in gall_imnames]
        atk_idens = ['_'.join(n.split('_')[:-1]) for n in atk_imnames]
        gall_idens = np.array([gall_idens])
        atk_idens = np.array([atk_idens])

        gall_imnames = np.array([gall_imnames])  # [1, 3166]
        atk_probe_imnames = np.array([atk_imnames])  # [1, 200]
        mask_exc = (atk_probe_imnames.T == gall_imnames) # [200, 3166]
        mask_type2 = (atk_idens.T == gall_idens)     # [200, 3166]
        mask_type2 &= ~mask_exc                   # exclude type1 attacks

        mask_type1 = (atk_idens.T == gall_idens)     # [200, 3166]

        # compute cosine similarity
        scores = cosine_similarity(atk_feats, gall_feats)

        inc_type1 = torch.zeros(10)
        exc_type1 = torch.zeros(10)
        for i in range(scores.shape[0]):
            mask1, mask2 = mask_type1[i], mask_type2[i]
            v, idx = torch.topk(scores[i], k=10)
            for j, mask_i in enumerate(idx):
                if mask1[mask_i]:
                    inc_type1[j:] += 1
                    break
            for j, mask_i in enumerate(idx):
                if mask2[mask_i]:
                    exc_type1[j:] += 1
                    break
        inc_type1 /= scores.shape[0]
        exc_type1 /= scores.shape[0]

        return inc_type1.numpy(), exc_type1.numpy()

    @torch.no_grad()
    def _get_top1_sample(self, args):
        img_names, img_dirs = dataset_parser(args, self.src_encoder_name)
        top1_dir_list = []
        imname_list = []    
        ## forward swa size 1 images and get top 1 index for each target IDs
        for i in tqdm(range(len(img_dirs))):
            ## img_names is like Ahmed_Chalabi_0001.jpg
            ## img_dirs is full directory like 'full_dir/--'/Ahmed_Chalabi_0001.jpg
            real_img_name = img_names[i]
            real_img_dir = img_dirs[i]

            ## check this part if the index -> directory is passed as designed
            atk_img_dirs = [os.path.join(self.atk_img_dir, real_img_name.split('.')[0], x) for x in sorted(os.listdir(os.path.join(self.atk_img_dir, real_img_name.split('.')[0])))]
            atk_img_dirs = atk_img_dirs[:args.num_latents]
            atk_imgs = []
            for atk_img_dir in atk_img_dirs:
                atk_img = self.trf(Image.open(atk_img_dir))
                atk_img = atk_img.to(args.device)
                atk_imgs.append(atk_img)
            atk_imgs = torch.stack(atk_imgs, dim=0)

            src_real_img = self.trf(Image.open(real_img_dir))
            src_real_img = src_real_img.unsqueeze(0).to(args.device)

            with torch.no_grad():
                feat_real = self.src_encoder(src_real_img, flip=args.flip)
                feat_atks = self.src_encoder(atk_imgs, flip=args.flip)

            cos_dist = cosine_similarity(feat_atks, feat_real)

            if args.use_val:
                _, src_idx = torch.topk(cos_dist.squeeze(), k=args.val_topk)
                val_img = atk_imgs[src_idx]

                tgt_val = self.val_encoder(val_img, flip=args.flip)
                mean_val = torch.mean(tgt_val, dim=0, keepdim=True)
                cos_val = cosine_similarity(tgt_val, mean_val)
                _, val_idx = cos_val.view(-1).max(0)
                idx_max = src_idx[val_idx]

            else:
                _, idx_max = cos_dist.max(0)

            dir_name = atk_img_dirs[idx_max.item()]
            top1_dir_list.append(dir_name)
            imname_list.append(real_img_name)

        return top1_dir_list, imname_list

    def _save_verification(self, atk_scores_type1, atk_scores_type2):
        res_array = np.zeros((3,4))
        for i, far_threshold_key in enumerate(['lfw_far_0.0001_threshold', 'lfw_far_0.001_threshold', 'lfw_far_0.01_threshold', 'lfw_threshold']):
            thresh = self.tgt_encoder_conf[far_threshold_key]
            res_array[0, i] = thresh
            res_array[1, i] = (atk_scores_type1 >= thresh).sum() / len(atk_scores_type1)
            res_array[2, i] = (atk_scores_type2 >= thresh).sum() / len(atk_scores_type2)
        
        for i in range(3):
            for j in range(4):
                if i != 0:
                    res_array[i, j] *= 100  # use % except for threshold
                res_array[i, j] = "{:.2f}".format(res_array[i, j])
        
        columns = ['0.0001', '0.0010', '0.0100', 'Acc']
        rows = ['Threshold', 'Type-1', 'Type-2']
        df = pd.DataFrame(res_array, rows, columns)
        df.to_excel(os.path.join(self.save_dir, f'{self.tgt_encoder_name}_verification.xlsx'))

    def _save_identification(self, inc_type1, exc_type1):
        result = np.concatenate((inc_type1.reshape(1, -1), exc_type1.reshape(1,-1)), axis=0) * 100
        df = pd.DataFrame(result)
        df.to_excel(os.path.join(self.save_dir, f'{self.tgt_encoder_name}_identification.xlsx'))




class cfp_evaluator():
    def __init__(self, args):
        self.trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.device = args.device
        self.save_dir = args.save_dir

        ## encoder used for generation
        self.src_encoder_name = args.src_encoder
        self.src_encoder = fetch_encoder(self.src_encoder_name).to(args.device)
        self.src_encoder = BlackboxEncoder(self.src_encoder, img_size=self.src_encoder.img_size)
        self.src_encoder.eval()

        ## encoder to attack
        self.tgt_encoder_name = args.tgt_encoder
        self.tgt_encoder = fetch_encoder(self.tgt_encoder_name).to(args.device)
        self.tgt_encoder = BlackboxEncoder(self.tgt_encoder, img_size=self.tgt_encoder.img_size)
        self.tgt_encoder.eval()

        self.val_encoder_name = args.val_encoder
        self.val_encoder = fetch_encoder(self.val_encoder_name).to(args.device)
        self.val_encoder = BlackboxEncoder(self.val_encoder, img_size=self.val_encoder.img_size)
        self.val_encoder.eval()

        self.atk_img_dir = os.path.join(args.save_dir, 'attack_images')


        with open('./config/encoder_config.yaml') as fp:
            self.tgt_encoder_conf = yaml.load(fp, Loader=yaml.FullLoader)
            self.tgt_encoder_conf = self.tgt_encoder_conf[args.tgt_encoder]

        with open('./config/dataset_config.yaml') as fp:
            self.dataset_conf = yaml.load(fp, Loader=yaml.FullLoader)
            self.dataset_conf = self.dataset_conf[args.dataset]


        self.root_dir = f'./dataset/cfp/Data/{args.align}_aligned'

        with open('./dataset/cfp/Protocol/Pair_list_F.txt','r') as fp:
            Flines = fp.readlines()
        with open('./dataset/cfp/Protocol/Pair_list_P.txt','r') as fp:
            Plines = fp.readlines()

        self.Fdict = {}
        self.F_sample2id = {}
        for line in Flines:
            num, path = line.strip().split()
            plist = path.split('/')
            plist[2] = f'{args.align}_aligned'
            path = '/'.join(plist)
            self.Fdict[num] = './dataset/cfp/Protocol/'+path
            id = path.split('/')[-3]
            self.F_sample2id[num] = id

        self.Pdict = {}
        self.P_sample2id = {}
        for line in Plines:
            num, path = line.strip().split()
            plist = path.split('/')
            plist[2] = f'{args.align}_aligned'
            path = '/'.join(plist)
            self.Pdict[num] = './dataset/cfp/Protocol/'+path


    @torch.no_grad()
    def verification(self, args):
        atk_dirs, _ = self._get_top1_sample(args)
        atk_scr_list = []
        for atk_dir in atk_dirs:
            k = atk_dir.split('/')[-2]
            imgdir = self.Fdict[k]
            img_atk = Image.open(atk_dir)
            img_atk = self.trf(img_atk).unsqueeze(0).cuda()

            img_ps = []
            base_dir = os.path.dirname(imgdir).replace('frontal', 'profile')
            for fname in sorted(os.listdir(base_dir)):
                imgdir_p = os.path.join(base_dir, fname)
                img_p = Image.open(imgdir_p)
                img_p = self.trf(img_p)
                img_ps.append(img_p)
            img_ps = torch.stack(img_ps).cuda()

            fea_atk = self.tgt_encoder(img_atk, self.flip)
            feas_p = self.tgt_encoder(img_ps, self.flip)

            atk_scr = cosine_similarity(feas_p, fea_atk)
            get_vals= atk_scr.cpu().numpy()
            atk_scr_list.extend(get_vals)

        atk_scores_type2 = np.array(atk_scr_list).squeeze()
        self._save_verification(atk_scores_type2)     

    @torch.no_grad()
    def identification(self, args):
        atk_dirs, atk_idens = self._get_top1_sample(args)
        gall_feats, gall_idens = self._get_feats(args=args, 
                                                 encoder=self.tgt_encoder,
                                                 type='gallery',
                                                 flip=self.flip)
        atk_feats, atk_idens = self._get_feats(args=args, 
                                      encoder=self.tgt_encoder,
                                        type='attack',
                                        atk_dirs=atk_dirs,
                                        atk_imnames= atk_idens)
        cmc = self._get_iden_atk_scores(gall_feats, gall_idens, atk_feats, atk_idens)
        self._save_identification(cmc)


    @torch.no_grad()
    def _get_feats(self, args, encoder, type, atk_dirs=None, atk_imnames=None, flip=True):
        encoder.eval()
        if type == 'attack':
            assert atk_dirs
            assert atk_imnames
            features = torch.FloatTensor([])
            idens = []
            for img_dir, imname in zip(atk_dirs, atk_imnames):
                assert imname.split('.')[0] in img_dir
                with torch.no_grad():
                    img = self.trf(Image.open(img_dir))
                    img = img.unsqueeze(0).to(args.device)
                    feat = encoder(img, flip=True).cpu()  # target feature
                    features = torch.cat((features, feat), dim=0)
                    id = self.F_sample2id[img_dir.split('/')[-2]]
                    idens.append(id)
            
        elif type == 'gallery':
            gall_dirs = []
            for name in os.listdir(os.path.join(self.root_dir)):
                gall_dirs += glob.glob(os.path.join(self.root_dir, name, 'profile', '*'))
            gall_set = cfp_dataset(gall_dirs, self.trf)

            gall_loader = DataLoader(gall_set, batch_size=60, shuffle=False, num_workers=2)
            features = torch.FloatTensor([])
            idens = []
            for _, (img, name) in tqdm(enumerate(gall_loader)):
                img = img.to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
                idens += list(name)
        return features, idens


    def _get_top1_sample(self, args):
        img_names, img_dirs = dataset_parser(args, self.src_encoder_name)
        top1_dir_list = []
        imname_list = []    
        ## forward swa size 1 images and get top 1 index for each target IDs
        for i in tqdm(range(len(img_dirs))):
            ## img_names is like Ahmed_Chalabi_0001.jpg
            ## img_dirs is full directory like 'full_dir/--'/Ahmed_Chalabi_0001.jpg
            real_img_name = img_names[i]
            real_img_dir = img_dirs[i]

            ## check this part if the index -> directory is passed as designed
            atk_img_dirs = [os.path.join(self.atk_img_dir, real_img_name.split('.')[0], x) for x in sorted(os.listdir(os.path.join(self.atk_img_dir, real_img_name.split('.')[0])))]
            atk_img_dirs = atk_img_dirs[:args.num_latents]
            atk_imgs = []
            for atk_img_dir in atk_img_dirs:
                atk_img = self.trf(Image.open(atk_img_dir))
                atk_img = atk_img.to(args.device)
                atk_imgs.append(atk_img)
            atk_imgs = torch.stack(atk_imgs, dim=0)

            src_real_img = self.trf(Image.open(real_img_dir))
            src_real_img = src_real_img.unsqueeze(0).to(args.device)

            with torch.no_grad():
                feat_real = self.src_encoder(src_real_img, flip=args.flip)
                feat_atks = self.src_encoder(atk_imgs, flip=args.flip)

            cos_dist = cosine_similarity(feat_atks, feat_real)

            if args.use_val:
                _, src_idx = torch.topk(cos_dist.squeeze(), k=args.val_topk)
                val_img = atk_imgs[src_idx]

                tgt_val = self.val_encoder(val_img, flip=args.flip)
                mean_val = torch.mean(tgt_val, dim=0, keepdim=True)
                cos_val = cosine_similarity(tgt_val, mean_val)
                _, val_idx = cos_val.view(-1).max(0)
                idx_max = src_idx[val_idx]

            else:
                _, idx_max = cos_dist.max(0)

            dir_name = atk_img_dirs[idx_max.item()]
            top1_dir_list.append(dir_name)
            imname_list.append(real_img_name)

        return top1_dir_list, imname_list


    @torch.no_grad()
    def _get_iden_atk_scores(self, gall_feats, gall_idens, atk_feats, atk_idens):
        gall_idens = np.array([gall_idens])
        atk_idens = np.array([atk_idens])

        masks = (atk_idens.T == gall_idens) # [200, 3166]

        # compute cosine similarity
        scores = cosine_similarity(atk_feats, gall_feats)

        cmc = torch.zeros(10)
        for i in range(scores.shape[0]):
            mask = masks[i]
            v, idx = torch.topk(scores[i], k=10)
            for j, mask_i in enumerate(idx):
                if mask[mask_i]:
                    cmc[j:] += 1
                    break
        cmc /= scores.shape[0]
        return cmc.numpy()


    def _save_verification(self, atk_scores_type2):
        res_array = np.zeros((2,4))
        for i, far_threshold_key in enumerate(['cfp_far_0.0001_threshold', 'cfp_far_0.001_threshold', 'cfp_far_0.01_threshold', 'lfw_threshold']):
            thresh = self.tgt_encoder_conf[far_threshold_key]
            res_array[0, i] = thresh
            res_array[1, i] = (atk_scores_type2 >= thresh).sum() / len(atk_scores_type2)
        
        for i in range(2):
            for j in range(4):
                if i != 0:
                    res_array[i, j] *= 100  # use % except for threshold
                res_array[i, j] = "{:.2f}".format(res_array[i, j])
        
        columns = ['0.0001', '0.0010', '0.0100', 'Acc']
        rows = ['Threshold', 'Type-2']
        df = pd.DataFrame(res_array, rows, columns)
        df.to_excel(os.path.join(self.save_dir, f'{self.tgt_encoder_name}_verification.xlsx'))


    def _save_identification(self, cmc):
        result = cmc.reshape(1,-1) * 100
        df = pd.DataFrame(result)
        df.to_excel(os.path.join(self.save_dir, f'{self.tgt_encoder_name}_identification.xlsx'))




class agedb_30_evaluator():
    def __init__(self, args):
        self.trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.device = args.device
        self.save_dir = args.save_dir

        ## encoder used for generation
        self.src_encoder_name = args.src_encoder
        self.src_encoder = fetch_encoder(self.src_encoder_name).to(args.device)
        self.src_encoder = BlackboxEncoder(self.src_encoder, img_size=self.src_encoder.img_size)
        self.src_encoder.eval()

        ## encoder to attack
        self.tgt_encoder_name = args.tgt_encoder
        self.tgt_encoder = fetch_encoder(self.tgt_encoder_name).to(args.device)
        self.tgt_encoder = BlackboxEncoder(self.tgt_encoder, img_size=self.tgt_encoder.img_size)
        self.tgt_encoder.eval()

        self.val_encoder_name = args.val_encoder
        self.val_encoder = fetch_encoder(self.val_encoder_name).to(args.device)
        self.val_encoder = BlackboxEncoder(self.val_encoder, img_size=self.val_encoder.img_size)
        self.val_encoder.eval()

        self.atk_img_dir = os.path.join(args.save_dir, 'attack_images')

        with open('./config/encoder_config.yaml') as fp:
            self.tgt_encoder_conf = yaml.load(fp, Loader=yaml.FullLoader)
            self.tgt_encoder_conf = self.tgt_encoder_conf[args.tgt_encoder]

        with open('./config/dataset_config.yaml') as fp:
            self.dataset_conf = yaml.load(fp, Loader=yaml.FullLoader)
            self.dataset_conf = self.dataset_conf[args.dataset]


    @torch.no_grad()
    def verification(self, args):
        atk_dirs, atk_imnames = self._get_top1_sample(args)
        pos_feats, pos_imnames = self._get_feats(args=args, 
                                                 encoder=self.tgt_encoder,
                                                 type='positive',
                                                 flip=self.flip)
        atk_feats, atk_imnames = self._get_feats(args=args,
                                        encoder=self.tgt_encoder,
                                        type='attack',
                                        atk_dirs=atk_dirs,
                                        atk_imnames= atk_imnames)
        atk_scores_type1, atk_scores_type2 = self._get_ver_atk_scores(pos_feats, pos_imnames, atk_feats, atk_imnames)
        self._save_verification(atk_scores_type1, atk_scores_type2)     

    @torch.no_grad()
    def identification(self, args):
        atk_dirs, atk_imnames = self._get_top1_sample(args)
        gall_feats, gall_imnames = self._get_feats(args=args, 
                                                 encoder=self.tgt_encoder,
                                                 type='gallery',
                                                 flip=self.flip)
        atk_feats, atk_imnames = self._get_feats(args=args, 
                                      encoder=self.tgt_encoder,
                                        type='attack',
                                        atk_dirs=atk_dirs,
                                        atk_imnames= atk_imnames)
        inc_type1, exc_type1 = self._get_iden_atk_scores(gall_feats, gall_imnames, atk_feats, atk_imnames)
        self._save_identification(inc_type1, exc_type1)

    @torch.no_grad()
    def _get_feats(self, args, encoder, type, atk_dirs=None, atk_imnames=None, flip=True):
        img_dir = self.dataset_conf['image_dir']+ f'/{args.align}_aligned'
        targets_txt = self.dataset_conf['targets_txt']
        with open(targets_txt, 'r') as fp:
            lines = fp.readlines()
        if type == 'positive':
            pos_ids = [l.strip().split('_')[1] for l in lines]
            pos_dirs = []
            for name in pos_ids:
                pos_dirs += glob.glob(os.path.join(img_dir, name, '*'))
            pos_set = agedb_30_dataset(pos_dirs, self.trf)
            pos_loader = DataLoader(pos_set, batch_size=60, shuffle=False, num_workers=2)
            features = torch.FloatTensor([])
            imnames = []
            for _, (img, name) in tqdm(enumerate(pos_loader)):
                img = img.to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
                imnames += list(name)
        elif type == 'attack':
            assert atk_dirs
            assert atk_imnames
            features = torch.FloatTensor([])
            imnames = []
            for dir in atk_dirs:
                imname = dir.split('/')[-2]+'.jpg'
                imnames.append(imname)
                img = self.trf(Image.open(dir))
                img = img.unsqueeze(0).to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)

        elif type == 'gallery':
            all_ids = os.listdir(img_dir)
            gall_dirs = []
            for name in all_ids:
                gall_dirs += glob.glob(os.path.join(img_dir, name, '*'))
            gall_set = lfw_dataset(gall_dirs, self.trf)
            gall_loader = DataLoader(gall_set, batch_size=60, shuffle=False, num_workers=2)

            features = torch.FloatTensor([])
            imnames = []
            for i, (img, name) in tqdm(enumerate(gall_loader)):
                img = img.to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
                imnames += list(name)
        else:
            raise ValueError(f'{type} is not a valid type; ["positive","negative","attack"]')

        return features, imnames


    @torch.no_grad()
    def _get_ver_atk_scores(self, pos_feats, pos_imnames, atk_feats, atk_imnames):
        pos_idens = ['_'.join(n.split('_')[:-3]) for n in pos_imnames]
        atk_idens = ['_'.join(n.split('_')[:-3]) for n in atk_imnames]
        pos_idens = np.array([pos_idens])
        atk_idens = np.array([atk_idens])
        pos_imnames = np.array([pos_imnames])  # [1, 3166]
        atk_imnames = np.array([atk_imnames])  # [1, 200]
        mask_type1 = (atk_imnames.T == pos_imnames) # [200, 3166]
        mask_type2 = (atk_idens.T == pos_idens)     # [200, 3166]
        mask_type2 &= ~mask_type1                   # exclude type1 attacks

        # compute cosine similarity
        scores = cosine_similarity(atk_feats, pos_feats)

        scores_type1 = torch.FloatTensor([])
        scores_type2 = torch.FloatTensor([])
        for i in range(scores.shape[0]):
            mask1, mask2 = mask_type1[i], mask_type2[i]
            scores_type1 = torch.cat((scores_type1, scores[i][mask1]), dim=0)
            scores_type2 = torch.cat((scores_type2, scores[i][mask2]), dim=0)

        return scores_type1.numpy(), scores_type2.numpy()


    @torch.no_grad()
    def _get_iden_atk_scores(self, gall_feats, gall_imnames, atk_feats, atk_imnames):
        gall_idens = ['_'.join(n.split('_')[:-3]) for n in gall_imnames]
        atk_idens = ['_'.join(n.split('_')[:-3]) for n in atk_imnames]
        gall_idens = np.array([gall_idens])
        atk_idens = np.array([atk_idens])

        gall_imnames = np.array([gall_imnames])  # [1, 3166]
        atk_imnames = np.array([atk_imnames])  # [1, 200]
        mask_exc = (atk_imnames.T == gall_imnames) # [200, 3166]
        mask_type2 = (atk_idens.T == gall_idens)     # [200, 3166]
        mask_type2 &= ~mask_exc                   # exclude type1 attacks

        mask_type1 = (atk_idens.T == gall_idens)     # [200, 3166]

        # compute cosine similarity
        scores = cosine_similarity(atk_feats, gall_feats)

        inc_type1 = torch.zeros(10)
        exc_type1 = torch.zeros(10)
        for i in range(scores.shape[0]):
            mask1, mask2 = mask_type1[i], mask_type2[i]
            v, idx = torch.topk(scores[i], k=10)
            for j, mask_i in enumerate(idx):
                if mask1[mask_i]:
                    inc_type1[j:] += 1
                    break
            for j, mask_i in enumerate(idx):
                if mask2[mask_i]:
                    exc_type1[j:] += 1
                    break
        inc_type1 /= scores.shape[0]
        exc_type1 /= scores.shape[0]

        return inc_type1.numpy(), exc_type1.numpy()

    @torch.no_grad()
    def _get_top1_sample(self, args):
        img_names, img_dirs = dataset_parser(args, self.src_encoder_name)
        top1_dir_list = []
        imname_list = []    
        ## forward swa size 1 images and get top 1 index for each target IDs
        for i in tqdm(range(len(img_dirs))):
            ## img_names is like Ahmed_Chalabi_0001.jpg
            ## img_dirs is full directory like 'full_dir/--'/Ahmed_Chalabi_0001.jpg
            real_img_name = img_names[i]
            real_img_dir = img_dirs[i]

            ## check this part if the index -> directory is passed as designed
            atk_img_dirs = [os.path.join(self.atk_img_dir, real_img_name.split('.')[0], x) for x in sorted(os.listdir(os.path.join(self.atk_img_dir, real_img_name.split('.')[0])))]
            atk_img_dirs = atk_img_dirs[:args.num_latents]
            atk_imgs = []
            for atk_img_dir in atk_img_dirs:
                atk_img = self.trf(Image.open(atk_img_dir))
                atk_img = atk_img.to(args.device)
                atk_imgs.append(atk_img)
            atk_imgs = torch.stack(atk_imgs, dim=0)

            src_real_img = self.trf(Image.open(real_img_dir))
            src_real_img = src_real_img.unsqueeze(0).to(args.device)

            with torch.no_grad():
                feat_real = self.src_encoder(src_real_img, flip=args.flip)
                feat_atks = self.src_encoder(atk_imgs, flip=args.flip)

            cos_dist = cosine_similarity(feat_atks, feat_real)

            if args.use_val:
                _, src_idx = torch.topk(cos_dist.squeeze(), k=args.val_topk)
                val_img = atk_imgs[src_idx]

                tgt_val = self.val_encoder(val_img, flip=args.flip)
                mean_val = torch.mean(tgt_val, dim=0, keepdim=True)
                cos_val = cosine_similarity(tgt_val, mean_val)
                _, val_idx = cos_val.view(-1).max(0)
                idx_max = src_idx[val_idx]

            else:
                _, idx_max = cos_dist.max(0)

            dir_name = atk_img_dirs[idx_max.item()]
            top1_dir_list.append(dir_name)
            imname_list.append(real_img_name)

        return top1_dir_list, imname_list

    def _save_verification(self, atk_scores_type1, atk_scores_type2):
        res_array = np.zeros((3,4))
        for i, far_threshold_key in enumerate(['agedb_30_far_0.0001_threshold', 'agedb_30_far_0.001_threshold', 'agedb_30_far_0.01_threshold', 'lfw_threshold']):
            thresh = self.tgt_encoder_conf[far_threshold_key]
            res_array[0, i] = thresh
            res_array[1, i] = (atk_scores_type1 >= thresh).sum() / len(atk_scores_type1)
            res_array[2, i] = (atk_scores_type2 >= thresh).sum() / len(atk_scores_type2)
        
        for i in range(3):
            for j in range(4):
                if i != 0:
                    res_array[i, j] *= 100  # use % except for threshold
                res_array[i, j] = "{:.2f}".format(res_array[i, j])
        
        columns = ['0.0001', '0.0010', '0.0100', 'Acc']
        rows = ['Threshold', 'Type-1', 'Type-2']
        df = pd.DataFrame(res_array, rows, columns)
        df.to_excel(os.path.join(self.save_dir, f'{self.tgt_encoder_name}_verification.xlsx'))

    def _save_identification(self, inc_type1, exc_type1):
        result = np.concatenate((inc_type1.reshape(1, -1), exc_type1.reshape(1,-1)), axis=0) * 100
        df = pd.DataFrame(result)
        df.to_excel(os.path.join(self.save_dir, f'{self.tgt_encoder_name}_identification.xlsx'))

