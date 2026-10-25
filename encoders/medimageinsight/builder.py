import os
import sys
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
import yaml
import torch
import torch.nn as nn
from torchvision.transforms import v2 as tfs
from warnings import warn
from safetensors.torch import load_file

from encoders.medimageinsight.vision_encoder import DaViT
from encoders.medimageinsight.lang_encoder import Transformer
from local_constants import (
    CLIP_VIT_BASE_PATCH32_CKPT,
    MII_CONFIG,
    MII_TEXT_TOKENIZER,
    MII_UNICL_CKPT,
    MII_VISION_CKPT,
)




def load_config_dict_to_opt(opt, config_dict, splitter='.', log_new=False):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split(splitter)
        pointer = opt
        for k_part in k_parts[:-1]:
            if '[' in k_part and ']' in k_part:
                # for the format "a.b[0][1].c: d"
                k_part_splits = k_part.split('[')
                k_part = k_part_splits.pop(0)
                pointer = pointer[k_part]
                for i in k_part_splits:
                    assert i[-1] == ']'
                    pointer = pointer[int(i[:-1])]
            else:
                if k_part not in pointer:
                    pointer[k_part] = {}
                pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        if '[' in k_parts[-1] and ']' in k_parts[-1]:
            k_part_splits = k_parts[-1].split('[')
            k_part = k_part_splits.pop(0)
            pointer = pointer[k_part]
            for i in k_part_splits[:-1]:
                assert i[-1] == ']'
                pointer = pointer[int(i[:-1])]
            assert k_part_splits[-1][-1] == ']'
            ori_value = pointer[int(k_part_splits[-1][:-1])]
            pointer[int(k_part_splits[-1][:-1])] = v
        else:
            ori_value = pointer.get(k_parts[-1])
            pointer[k_parts[-1]] = v
        if ori_value:
            warn(f"Overrided {k} from {ori_value} to {v}")
        elif log_new:
            warn(f"Added {k}: {v}")
            
            

def load_opt_from_config_files(conf_files):
    """
    Load opt from the config files, settings in later files can override those in previous files.
    Args:
        conf_files (list): a list of config file paths
    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    if not isinstance(conf_files, (list, tuple)):
        conf_files = [conf_files]
        
    for conf_file in conf_files:
        with open(conf_file, encoding='utf-8') as f:
            # config_dict = yaml.safe_load(f)
            config_dict = yaml.unsafe_load(f)

        load_config_dict_to_opt(opt, config_dict)

    return opt

def post_process_clip(text):
    text['input_ids'].squeeze_() # torch.Size([1, 77])
    text['attention_mask'].squeeze_() # torch.Size([1, 77])
    return text


def build_tokenizer(config_encoder):
    from transformers import CLIPTokenizer
    assert config_encoder['TOKENIZER'] == 'clip', f"Check tokenizer in MedImageInsight: {config_encoder['TOKENIZER']}"
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    pretrained_tokenizer = config_encoder.get(
        'PRETRAINED_TOKENIZER', CLIP_VIT_BASE_PATCH32_CKPT
    )
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
    tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    tokenizer.post_process = post_process_clip
    return tokenizer


def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    transformer = Transformer(
        context_length=config_encoder['CONTEXT_LENGTH'],
        vocab_size=tokenizer.vocab_size,
        width=config_encoder['WIDTH'],
        layers=config_encoder['LAYERS'],
        heads=config_encoder['HEADS'],
        autogressive=config_encoder.get('AUTOGRESSIVE', True),
        key_padding_token=config_encoder.get('KEY_PADDING_TOKEN', 0),
    )

    if config_encoder['LOAD_PRETRAINED']:
        transformer.load_pretrained()

    return transformer


def build_image_encoder(config_encoder, img_size=None):
    spec = config_encoder['SPEC']
    if img_size is None:
        img_size = config_encoder.get('IMAGE_SIZE', 480)

    # print(f">>> config_encoder: {config_encoder}")
    # print(f">>> spec: {spec}")
    
    model = DaViT(
        num_classes=config_encoder['NUM_CLASSES'],
        depths=spec['DEPTHS'],
        embed_dims=spec['DIM_EMBED'],
        base_embed_dims=spec['DIM_EMBED'],
        num_heads=spec['NUM_HEADS'],
        base_num_heads=spec['NUM_HEADS'],
        num_groups=spec['NUM_GROUPS'],
        base_num_groups=spec['NUM_GROUPS'],
        patch_size=spec['PATCH_SIZE'],
        patch_stride=spec['PATCH_STRIDE'],
        patch_padding=spec['PATCH_PADDING'],
        patch_prenorm=spec['PATCH_PRENORM'],
        drop_path_rate=spec['DROP_PATH_RATE'],
        img_size=img_size,  # This argument is actually no use in the architecture.
        window_size=spec.get('WINDOW_SIZE', 7),
        enable_checkpoint=spec.get('ENABLE_CHECKPOINT', False),
        conv_at_attn=spec.get('CONV_AT_ATTN', True),
        conv_at_ffn=spec.get('CONV_AT_FFN', True),
        dynamic_scale=spec.get('DYNAMIC_SCALE', True),
        standparam=True,
    )

    return model


class UniCLModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.conf_lang_encoder = config['LANG_ENCODER']
        self.tokenizer = build_tokenizer(self.conf_lang_encoder)
        self.lang_encoder = build_lang_encoder(self.conf_lang_encoder, self.tokenizer, config['VERBOSE'])

        dim_projection = config['UNICL_MODEL']['DIM_PROJECTION']
        if hasattr(self.lang_encoder, 'dim_out'):
            dim_out = self.lang_encoder.dim_out
        else:
            with torch.no_grad():
                dim_out = self.lang_encoder(
                    torch.zeros(1,1).type(torch.LongTensor)
                )['last_hidden_state'].size(2)

        self.lang_projection = nn.Parameter(torch.empty(dim_out, dim_projection))

        self.conf_image_encoder = config['IMAGE_ENCODER']
        self.image_encoder = build_image_encoder(self.conf_image_encoder)

        self.image_projection = nn.Parameter(
            torch.empty(self.image_encoder.dim_out, dim_projection)
        )

        self.logit_scale = nn.Parameter(torch.ones([]))

        if torch.cuda.is_available():
            self.device = torch.device(type="cuda", index=0)
        else:
            self.device = torch.device(type="cpu")

    def custom_init_weights(self, use_original_init=True):
        self.use_original_init = use_original_init
        nn.init.trunc_normal_(self.lang_projection, std=.02)
        nn.init.trunc_normal_(self.image_projection, std=.02)

    def _convert_old_weights(self, model_dict):
        model_dict_updated = {}
        for k, v in model_dict.items():
            if k.startswith('visual.'):
                model_dict_updated['image_encoder.'+k[7:]] = v
            elif k.startswith('text.'):
                model_dict_updated['lang_encoder.'+k[5:]] = v
            elif k == 'vision_projection':
                model_dict_updated['image_projection'] = v
            elif k == 'text_projection':
                model_dict_updated['lang_projection'] = v
            else:
                model_dict_updated[k] = v

        return model_dict_updated

    def from_pretrained(self, pretrained='', pretrained_layers=[], verbose=True):
        if not os.path.isfile(pretrained):
            print(f'=> Pretrained model ({pretrained}) is not a file, skip init weight')
            return

        ## Load SafeTensors Version of Pretrained Model
        pretrained_dict = load_file(pretrained)
        print(f'=> Loading pretrained model {pretrained}')
        model_dict = self.state_dict()
        pretrained_dict = self._convert_old_weights(pretrained_dict)
        ## To ensure cuda is mapped to all weights in the SafeTensors version model
        pretrained_dict = {
            k: v.to(self.device) for k, v in pretrained_dict.items()
        }
        need_init_state_dict = {}
        image_encoder_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = (
                k.split('.')[0] in pretrained_layers
                or pretrained_layers[0] == '*'
            )

            if need_init:
                if k.startswith('image_encoder.'):
                    image_encoder_state_dict[k] = v.to(self.device)
                else:
                    if verbose:
                        print(f'=> init {k} from {pretrained}')

                    if 'positional_embedding' in k and v.size() != model_dict[k].size():
                        positional_embedding_pretrained = v
                        positional_embedding_current = model_dict[k]
                        L1, nH1 = positional_embedding_pretrained.size()
                        L2, nH2 = positional_embedding_current.size()
                        if nH1 != nH2:
                            print(f"Error in loading {k}, passing")
                        else:
                            if L1 != L2:
                                print(
                                    '=> load_pretrained: resized variant: {} to {}'
                                        .format((L1, nH1), (L2, nH2))
                                )

                                posemb = positional_embedding_pretrained.float()
                                posemb_grid = posemb.unsqueeze(dim=0).permute(0, 2, 1)
                                posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=L2, mode='linear')
                                posemb_grid = posemb_grid.permute(0, 2, 1).squeeze(dim=0)
                                v = posemb_grid

                    need_init_state_dict[k] = v.to(self.device)
        self.image_encoder.from_state_dict(image_encoder_state_dict, ['*'], verbose)
        self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = {'logit_scale'}
        if hasattr(self.lang_encoder, 'no_weight_decay'):
            for k in self.lang_encoder.no_weight_decay():
                no_weight_decay.add('lang_encoder.'+k)

        if hasattr(self.image_encoder, 'no_weight_decay'):
            for k in self.image_encoder.no_weight_decay():
                no_weight_decay.add('image_encoder.'+k)

        return no_weight_decay

    @property
    def dtype(self):
        return self.logit_scale.dtype

    def encode_image(self, image, norm=True):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_text(self, text, norm=True):
        x = self.lang_encoder(**text)
        x = x['last_hidden_state']

        if self.conf_lang_encoder['TOKENIZER'] == 'clip':
            x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.lang_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        return features_image, features_text, T



def build_model(
    arch='medimageinsight-vision',
    image_shape=(480, 480),
    device='cpu',
    pretrained_path=None,
    pretrained_config=None,
    text_tokenizer_path=None,
    unicl_ckpt_path=None,
    vision_ckpt_path=None,
):
    pretrained_config = MII_CONFIG if pretrained_config is None else pretrained_config
    text_tokenizer_path = (
        MII_TEXT_TOKENIZER if text_tokenizer_path is None else text_tokenizer_path
    )
    unicl_ckpt_path = MII_UNICL_CKPT if unicl_ckpt_path is None else unicl_ckpt_path
    if vision_ckpt_path is None:
        vision_ckpt_path = pretrained_path or MII_VISION_CKPT

    if not isinstance(image_shape, (list, tuple)):
        image_shape = [image_shape, image_shape]

    assert len(image_shape) == 2, f"Should specify a tuple for `image_shape`, got {image_shape}"

    config = load_opt_from_config_files(pretrained_config)
    config['LANG_ENCODER']['PRETRAINED_TOKENIZER'] = text_tokenizer_path
    config['UNICL_MODEL']['PRETRAINED'] = unicl_ckpt_path

    if 'vision' in arch:
        if vision_ckpt_path is None:
            raise ValueError("vision_ckpt_path must be provided for vision-only MedImageInsight.")
        config_encoder = config['IMAGE_ENCODER']
        model = build_image_encoder(config_encoder)
        state_dict = torch.load(vision_ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    else:
        if unicl_ckpt_path is None:
            raise ValueError("unicl_ckpt_path must be provided for full UniCL MedImageInsight.")
        model = UniCLModel(config)
        model.custom_init_weights(use_original_init=True)
        model.from_pretrained(unicl_ckpt_path, config['UNICL_MODEL']['PRETRAINED_LAYERS'], config['VERBOSE'])

    return model.to(device)


def build_transform(image_shape=(480, 480)):
    return tfs.Compose([
        tfs.ToTensor(),
        tfs.Resize(image_shape, interpolation=3, antialias=True),
        tfs.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])



def get_base_dim(image_shape=(480, 480), *args, **kwargs):
    raw_dim = 2048  # right after the encoder
    latent_dim = 2048  # latent
    feature_dim = 2048  # the projected feature
    ds_factor = 32

    if isinstance(image_shape, (list, tuple)):
        feature_shape = [s // ds_factor for s in image_shape]
    else:
        feature_shape = [image_shape // ds_factor, image_shape // ds_factor]

    return {
        'raw_dim': raw_dim,
        'latent_dim': latent_dim,
        'feature_dim': feature_dim,
        'raw_shape': feature_shape,
        'latent_shape': feature_shape,
        'feature_shape': feature_shape
    }


if __name__ == '__main__':
    x = torch.randn((2, 3, 512, 512))
    model = build_model()
    with torch.no_grad():
        out = model(x)
    for k,v in out.items():
        print(f"{k}: {v.shape}")
