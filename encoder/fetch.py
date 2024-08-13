import os
import yaml
import torch
from encoder.hog_descriptor import HOG_descriptor, HOG_wrapper
from encoder.VGGNets import VGG
from encoder.ResNets import Resnet
from encoder.iresnet import iresnet100
from encoder.MobileFaceNet import MobileFaceNet
from encoder.Swin_Transformer import SwinTransformer
from encoder.AdaFace import build_model
from encoder.CurricularFace import IR_101
from encoder.ElasticFace import iresnet100

def fetch_encoder(encoder_type, pretrained=True,
                  encoder_conf_file=f"./config/encoder_config.yaml"):
    with open(encoder_conf_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        conf = conf[encoder_type]

    if encoder_type == "HOG":
        cell_size = conf['cell_size']
        bin_size = conf['bin_size']
        hog = HOG_descriptor(cell_size=cell_size, bin_size=bin_size)
        encoder = HOG_wrapper(hog)

    elif encoder_type == 'MobileFaceNet':
        feat_dim = conf['feat_dim']  # dimension of the output features, e.g. 512.
        out_h = conf['out_h']  # height of the feature map before the final features.
        out_w = conf['out_w']  # width of the feature map before the final features.
        encoder = MobileFaceNet(feat_dim, out_h, out_w)

    elif encoder_type == 'VGGNet19':
        encoder = VGG('VGG19')

    elif encoder_type == 'ResNet50':
        depth = conf['depth']  # depth of the ResNet, e.g. 50, 100, 152.
        drop_ratio = conf['drop_ratio']  # drop out ratio.
        net_mode = conf['net_mode']  # 'ir' for improved by resnt, 'ir_se' for SE-ResNet.
        feat_dim = conf['feat_dim']  # dimension of the output features, e.g. 512.
        out_h = conf['out_h']  # height of the feature map before the final features.
        out_w = conf['out_w']  # width of the feature map before the final features.
        encoder = Resnet(depth, drop_ratio, net_mode, feat_dim, out_h, out_w)

    elif encoder_type == 'ResNet100':
        encoder = iresnet100()

    elif encoder_type == 'ResNet152':
        depth = conf['depth']  # depth of the ResNet, e.g. 50, 100, 152.
        drop_ratio = conf['drop_ratio']  # drop out ratio.
        net_mode = conf['net_mode']  # 'ir' for improved by resnt, 'ir_se' for SE-ResNet.
        feat_dim = conf['feat_dim']  # dimension of the output features, e.g. 512.
        out_h = conf['out_h']  # height of the feature map before the final features.
        out_w = conf['out_w']  # width of the feature map before the final features.
        encoder = Resnet(depth, drop_ratio, net_mode, feat_dim, out_h, out_w)

    elif encoder_type == 'AdaFace':
        encoder = build_model('ir_101')

    elif encoder_type == 'CurricularFace':
        encoder = IR_101((112, 112))

    elif encoder_type == 'ElasticFace':
        encoder = iresnet100()

    elif encoder_type == 'Swin-S' or encoder_type == 'Swin-T':
        img_size = conf['img_size']
        patch_size = conf['patch_size']
        in_chans = conf['in_chans']
        embed_dim = conf['embed_dim']
        depths = conf['depths']
        num_heads = conf['num_heads']
        window_size = conf['window_size']
        mlp_ratio = conf['mlp_ratio']
        drop_rate = conf['drop_rate']
        drop_path_rate = conf['drop_path_rate']
        encoder = SwinTransformer(img_size=img_size,
                                   patch_size=patch_size,
                                   in_chans=in_chans,
                                   embed_dim=embed_dim,
                                   depths=depths,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=True,
                                   qk_scale=None,
                                   drop_rate=drop_rate,
                                   drop_path_rate=drop_path_rate,
                                   ape=False,
                                   patch_norm=True,
                                   use_checkpoint=False)
    elif encoder_type == 'FaceNet':
        from facenet_pytorch import InceptionResnetV1
        encoder = InceptionResnetV1(pretrained='vggface2')
    else:
        raise NotImplementedError(f"{encoder_type} is not implemented!")

    # save image size
    encoder.img_size = conf['img_size']

    # activate eval mode
    encoder.eval()

    if encoder_type == 'AdaFace':
        statedict = torch.load(conf['weight'], map_location='cpu')['state_dict']
        model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        encoder.load_state_dict(model_statedict)
    elif pretrained and encoder_type not in ['FaceNet', 'HOG']:
        stdict = torch.load(conf['weight'], map_location='cpu')
        encoder.load_state_dict(stdict)

    return encoder