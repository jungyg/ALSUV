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
from test_utils import lfw_evaluator, cfp_evaluator, agedb_30_evaluator


def generate(args):
    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    assert (args.num_latents//args.mini_batch) * args.mini_batch == args.num_latents

    # get attack targets
    targets, imgdirs = dataset_parser(args, args.src_encoder)
    num_targets = len(targets)

    # get generator
    stgan = StyleGANWrapper(args)

    # get target encoder
    enc_tgt = fetch_encoder(args.src_encoder)
    enc_tgt = WhiteboxEncoder(enc_tgt, img_size=enc_tgt.img_size).to(args.device)

    # standard image transform
    resize = transforms.Resize((args.crop_size, args.crop_size))
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # random samples for initialization
    if args.load_wp:
        w = np.load("random_init_samples_10k.npy")
    else:
    # load random W space latents from pre-trained StyleGAN2
        with torch.no_grad():
            z = torch.randn(args.num_anchors, 512).to(args.device)
            w = stgan.generator.style(z)
        np.save('./random_init_samples_10k.npy', w)

    w = w[:args.num_latents]
    w = torch.from_numpy(w).to(args.device)
    wp = w.unsqueeze(1).repeat(1, stgan.generator.n_latent, 1)

    # Attack(Generate) dataset
    target_list = os.listdir(os.path.join(args.save_dir, 'attack_images'))
    for cnt in range(num_targets):
        target = targets[cnt]
        if target.split('.')[0] in target_list:
            continue
        imgdir = imgdirs[cnt]
        print(f"{cnt}: {target}, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        latent_seq = torch.FloatTensor().to(args.device)

        # align image & compute target feature
        with torch.no_grad():
            img = trf(Image.open(imgdir))
            img = img.unsqueeze(0).to(args.device)
            feat_tgt = enc_tgt(img, flip=True)  # target feature
        ## end of sampling 100 from 10k


        latent_all = wp.clone().view(args.num_latents//args.mini_batch, args.mini_batch, stgan.generator.n_latent, 512)
        latent_list = []
        for k in range(args.num_latents//args.mini_batch):
            # set optimizer
            latent = latent_all[k].clone()
            latent.requires_grad = True
            optimizer = optim.Adam([latent], lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=args.lr_gamma)

            latent_seq = torch.FloatTensor().to(args.device)

            # optimization
            for i in tqdm(range(1, args.iters+1)):
                ltn = latent  # match variable name
                img_gen = stgan(ltn)
                feat_gen = enc_tgt(img_gen, flip=args.flip)
                cos_gen = cosine_similarity(feat_gen, feat_tgt)
                loss = -cos_gen.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()


                ltn_save = latent.detach().clone().unsqueeze(0)
                latent_seq = torch.cat((latent_seq, ltn_save), dim=0)


            latent_list.append(latent_seq)

        ## get LA latent (latent_seq is shape LA_ITER X NUM_INIT X 512)
        for k in range(len(latent_list)):
            la_idx = args.la_size
            la_latent = latent_list[k][-la_idx:]
            la_latent = torch.mean(la_latent, dim=0)
            with torch.no_grad():
                img_gen = stgan(la_latent)
            for j in range(img_gen.shape[0]):
                idx_str = '0' * (4 - len(str((args.mini_batch*k)+j))) + str((args.mini_batch*k)+j)
                dir_name = target.split('.')[0]
                os.makedirs(f'{args.save_dir}/attack_images/{dir_name}', exist_ok=True)
                os.makedirs(os.path.join(args.save_dir, 'attack_images', dir_name), exist_ok=True)
                img_name = f'la_{la_idx}.jpg'
                utils.save_image(img_gen[j], os.path.join(args.save_dir, 'attack_images', dir_name, f'{idx_str}_{img_name}'), nrow=1, normalize=True, range=(-1, 1))



def test(args):
    if args.dataset=='lfw-200':
        evaluator = lfw_evaluator(args)
    elif args.dataset=='cfp-fp-200-F':
        evaluator = cfp_evaluator(args)
    elif args.dataset == 'agedb_30_200':
        evaluator = agedb_30_evaluator(args)

    evaluator.verification(args)
    evaluator.identification(args)



def parse_args():
    parser = argparse.ArgumentParser(description="test deep face reconstruction via Genetic Algorithm")

    parser.add_argument('--test', default=False, type=str2bool, help='test or train')

    ################################################################################################
    ################################################################################################
    ################################################################################################

    # Train args
    # optimization parameters
    parser.add_argument('--iters', default=100, type=int, help='number of iteration per attack')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate; cosine decay is used')
    parser.add_argument('--lr_decay', default=[50], type=list)
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--mini_batch', default=10, type=int, help='minibatch to prevent GPU out of memory')

    # ALSUV args
    parser.add_argument('--num_latents', default=100, type=int, help='number of latents to optimize')
    parser.add_argument('--la_size', default=70, type=int, help='size of latent average, 1 indicates without latent averaging')

    # source dataset & encoder
    parser.add_argument('--flip', default=True, type=str2bool, help='use horizontal flip as TTA')
    parser.add_argument('--dataset', default='lfw-200', type=str, help='target dataset to attack', choices=['lfw-200', 'cfp-fp-200-F', 'agedb_30_200'])
    parser.add_argument('--src_encoder', default='ResNet100', type=str, help='source encoder', choices=['VGGNet19', 'FaceNet', 'ResNet50', 'Swin-S', 'MobileFaceNet', 'ResNet100'])
    parser.add_argument('--dataset_target_idx', default='0,201', type=lambda s: [int(item) for item in s.split(',')], help='start and end index of target from dataset to generate')


    # StyleGAN & alignment model parameters - need not change
    parser.add_argument('--resolution', default=256, type=int, help='StyleGAN output resolution')
    parser.add_argument('--batch_size', default=50, type=int, help='StyleGAN batch size. Reduce to avoid OOM')
    parser.add_argument('--truncation', default=0.8, type=int, help='interpolation weight w.r.t. initial latent')
    parser.add_argument('--generator_type', default='FFHQ-256', type=str)
    parser.add_argument('--crop_size', default=192, type=int, help='crop size for StyleGAN output')
    parser.add_argument('--load_wp', default=True, type=str2bool, help='True loads saved latents, False initializes and save new latents')

    # Misc.
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--device_id', default=0, type=int, help='which gpu to use')


    ################################################################################################
    ################################################################################################
    ################################################################################################

    # Test args
    parser.add_argument('--tgt_encoder', default='MobileFaceNet', type=str, help='target encoder to attack', choices=['VGGNet19', 'FaceNet', 'ResNet50', 'Swin-S', 'MobileFaceNet', 'ResNet100'])

    # ALSUV args
    parser.add_argument('--use_val', default=True, type=str2bool, help='validation encoder flag')
    parser.add_argument('--val_encoder', default='Swin-T', type=str, help='type of validation encoder', choices=['Swin-T'])
    parser.add_argument('--val_topk', default=10, type=int, help='number of validation samples')




    args = parser.parse_args()
    args.device = f'cuda:{args.device_id}'
    # face alignment method
    if args.src_encoder in ['FaceNet', 'VGGNet19']:
        args.align = 'mtcnn'
    elif args.src_encoder in ['ResNet50', 'Swin-S', 'MobileFaceNet', 'Swin-T', 'ResNet100']:
        args.align = 'FXZoo'
    else:
        raise NotImplementedError(f"Alignment for {args.src_encoder} is not implemented!")

    if args.val_encoder in ['FaceNet', 'VGGNet19']:
        args.val_align = 'mtcnn'
    elif args.val_encoder in ['ResNet50', 'Swin-S', 'MobileFaceNet', 'Swin-T', 'ResNet100']:
        args.val_align = 'FXZoo'

    if 'agedb' in args.dataset:
        args.align = 'Insightface'
        args.val_align = 'Insightface'

    # make directory for saving results
    args.save_dir = f'./results/SrcEnc_{args.src_encoder}/{args.dataset}'
    os.makedirs(f'{args.save_dir}/attack_images', exist_ok=True)

    return args


if __name__ == "__main__":
    args = parse_args()

    # save arguments
    argdict = args.__dict__.copy()
    with open(f'{args.save_dir}/args.txt', 'w') as fp:
        json.dump(argdict, fp, indent=2)

    if args.test:
        test(args)
    else:
        generate(args)