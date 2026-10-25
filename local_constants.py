WEIGHT_ROOT = '/path/to/your/weights'

# General classifiers
CLIP_VIT_BASE_PATCH32_CKPT = f'{WEIGHT_ROOT}/clip-vit-base-patch32'
ALEXNET_CKPT = f'{WEIGHT_ROOT}/cls/alexnet-owt-7be5be79.pth'
SQUEEZENET_CKPT = f'{WEIGHT_ROOT}/cls/squeezenet1_1-b8a52dc0.pth'
VGG16_CKPT = f'{WEIGHT_ROOT}/cls/vgg16-397923af.pth'
INCEPTION_CKPT = f'{WEIGHT_ROOT}/fid/pt_inception-2015-12-05-6726825d.pth'
INCEPTIONW_CKPT = f'{WEIGHT_ROOT}/fid/inception-2015-12-05.pt'

# MedImageInsight
MII_UNICL_CKPT = f'{WEIGHT_ROOT}/medimageinsight/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt'
MII_VISION_CKPT = f'{WEIGHT_ROOT}/medimageinsight/pytorch_weights/image_encoder.pt'
MII_TEXT_TOKENIZER = f'{WEIGHT_ROOT}/medimageinsight/2024.09.27/language_model/clip_tokenizer_4.16.2'
MII_CONFIG = f'{WEIGHT_ROOT}/medimageinsight/2024.09.27/config.yaml'

# DermLIP
DERMLIP_CKPT = f'{WEIGHT_ROOT}/DermLIP'

# CONCH
CONCH_CKPT = f'{WEIGHT_ROOT}/CONCH'

# PeskaVLP
PESKAVLP_CKPT = f'{WEIGHT_ROOT}/PeskaVLP/PeskaVLP.pth'

# RETCLIP
RETCLIP_CKPT = f'{WEIGHT_ROOT}/retclip/ret-clip.pt'

# SigLIP2
SIGLIP2SO400M_CKPT = f'{WEIGHT_ROOT}/siglip2-so400m-patch16-256'

# HLIP
HLIPV_CKPT = f'{WEIGHT_ROOT}/HLIP/clip-vit_large-slice_scan_study-dualdinotxt1568/hlip_vision_weights.pth'

# SPECTRE
SPECTRE_BACKBONE_CKPT = f'{WEIGHT_ROOT}/SPECTRE/spectre_backbone_vit_large_patch16_128.pt'
SPECTRE_COMBINER_CKPT = f'{WEIGHT_ROOT}/SPECTRE/spectre_combiner_feature_vit_large.pt'
SPECTRE_PROJECTION_CKPT = f'{WEIGHT_ROOT}/SPECTRE/SigLIP_projection_head_image.pt'
