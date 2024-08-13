import os
import yaml


def dataset_parser(args, encoder_name):
    if encoder_name in ['FaceNet', 'VGGNet19', 'AdaFace', 'CurricularFace', 'ElasticFace']:
        align = 'mtcnn'
    elif encoder_name in ['ResNet50', 'Swin-S', 'MobileFaceNet', 'ResNet100']:
        align = 'FXZoo'
    if args.dataset == 'agedb_30':
        align = 'Insightface'

    with open('./config/dataset_config.yaml') as fp:
        conf = yaml.load(fp, Loader=yaml.FullLoader)
        conf = conf[args.dataset]

    img_dir = conf['image_dir']
    img_dir = img_dir + f'/{align}_aligned'
    targets_txt = conf['targets_txt']

    with open(targets_txt, 'r') as fp:
        lines = fp.readlines()
    target_list = [l.strip() for l in lines]
    target_list = target_list[args.dataset_target_idx[0]:args.dataset_target_idx[1]]
    print(f'Attack Dataset is {args.dataset} from index {args.dataset_target_idx[0]} : {target_list[0]} to {args.dataset_target_idx[1]} : {target_list[-1]}')

    if args.dataset == 'cfp-fp-F' or args.dataset == 'cfp-fp-P' or args.dataset == 'cfp-fp-200-F' or args.dataset == 'cfp-fp-200-P':
        mode = args.dataset[-1]  # F or P
        protocol_dir = conf['protocol_dir']
        with open(protocol_dir + f'/Pair_list_{mode}.txt', 'r') as fp:
            lines = fp.readlines()
        idx_dict = {}
        for line in lines:
            num, path = line.strip().split()
            plist = path.split('/')
            plist[2] = f'{align}_aligned'
            path = '/'.join(plist)
            idx_dict[num] = protocol_dir + '/' + path

    targets = []
    imgdirs = []
    for target in target_list:
        if args.dataset == 'lfw' or args.dataset == 'lfw-200':
            target_name = target.split('/')[-1][:-9]
            imgdir = os.path.join(img_dir, target_name, target)
        elif args.dataset == 'cfp-fp-F' or args.dataset == 'cfp-fp-P' or args.dataset == 'cfp-fp-200-F' or args.dataset == 'cfp-fp-200-P':
            imgdir = idx_dict[target]
            target = target
        elif args.dataset == 'colorferet-dup1' or args.dataset == 'colorferet-dup2':
            tokens = target.split(' ')
            imgdir = os.path.join(img_dir, tokens[0], tokens[1])
            target = tokens[1]
        elif args.dataset == 'agedb_30' or args.dataset=='agedb_30_200':
            target_name = target.split('_')[1]
            imgdir = os.path.join(img_dir, target_name, target)
        else:
            raise NotImplementedError(f'dataset {args.dataset} is not implemented!')
        targets.append(target)
        imgdirs.append(imgdir)

    return targets, imgdirs