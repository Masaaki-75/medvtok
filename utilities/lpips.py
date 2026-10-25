"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange



def tensor_list_to_batch(tensors, fix_size=256, fix_depth=8,):
    if isinstance(tensors, torch.Tensor):
        # Also ensure tensor shape for batched tensors, because 
        # we set tokenizer to output batched tensor when possible.
        # However, the input tensor may be given as a list.
        batch_size = tensors.shape[0]
        num_dims = len(tensors.shape)
        
        if num_dims == 4:
            # [B, C, H, W] -> [B, C, fix_size, fix_size]
            tensors = F.interpolate(tensors, (fix_size, fix_size), antialias=True, mode='bilinear')
        elif num_dims == 5:
            if tensors.shape[1] == 3 and tensors.shape[1] < min(tensors.shape[3:]):
                # [B, C, T, H, W]
                if fix_depth is not None and tensors.shape[2] > 3:
                    b = tensors.shape[0]
                    tensors = rearrange(tensors, 'b c t h w -> (b t) c h w')
                    tensors = F.interpolate(tensors, (fix_size, fix_size), antialias=True, mode='bilinear')
                    tensors = rearrange(tensors, '(b t) c h w -> (b h w) c t', b=b)
                    tensors = F.interpolate(tensors, size=fix_depth, mode='linear', align_corners=False)
                    tensors = rearrange(tensors, '(b h w) c t -> (b t) c h w', b=b, h=fix_size, w=fix_size)
                else:
                    tensors = rearrange(tensors, 'b c t h w -> (b t) c h w')
                    tensors = F.interpolate(tensors, (fix_size, fix_size), antialias=True, mode='bilinear')
            elif tensors.shape[2] == 3 and tensors.shape[2] < min(tensors.shape[3:]):
                # [B, T, C, H, W]
                if fix_depth is not None and tensors.shape[1] > 3:
                    b = tensors.shape[0]
                    tensors = rearrange(tensors, 'b t c h w -> (b t) c h w')
                    tensors = F.interpolate(tensors, (fix_size, fix_size), antialias=True, mode='bilinear')
                    tensors = rearrange(tensors, '(b t) c h w -> (b h w) c t', b=b)
                    tensors = F.interpolate(tensors, size=fix_depth, mode='linear', align_corners=False)
                    tensors = rearrange(tensors, '(b h w) c t -> (b t) c h w', b=b, h=fix_size, w=fix_size)
                else:
                    tensors = rearrange(tensors, 'b t c h w -> (b t) c h w')
                    tensors = F.interpolate(tensors, (fix_size, fix_size), antialias=True, mode='bilinear')
            else:
                raise ValueError(f"Invalid shape for a 5D batched input, expected [B, 3, T, H, W] or [B, T, 3, H, W], got {tensors.shape}")
        else:
            raise ValueError(f"Invalid tensor shapes in input batched tensor. Expected [B, 3, H, W] or [B, 3, T, H, W], got: {tensors.shape}")
    elif isinstance(tensors, (list, tuple)):
        batch_size = len(tensors)
        num_dims = len(tensors[0].shape)
        tensor_shapes = [tuple(t.shape) for t in tensors]
        example_shape = tensor_shapes[0]

        if num_dims == 3:  
            # B*[C, H, W] -> [B, C, fix_size, fix_size]
            tensors = torch.cat([
                F.interpolate(tensors[i].unsqueeze(0), (fix_size, fix_size), antialias=True, mode='bilinear')
            for i in range(batch_size)], dim=0)
        elif num_dims == 4:
            if example_shape[1] == 3:
                # B*[T, C, H, W]
                if fix_depth is not None and example_shape[0] > 3:
                    results = []
                    for i in range(batch_size):
                        t = F.interpolate(tensors[i], (fix_size, fix_size), antialias=True, mode='bilinear')
                        t = rearrange(t, 't c h w -> (h w) c t')
                        t = F.interpolate(t, size=fix_depth, mode='linear', align_corners=False)
                        results.append(rearrange(t, '(h w) c t -> t c h w', h=fix_size, w=fix_size))
                    tensors = torch.cat(results, dim=0)
                else:
                    tensors = torch.cat([
                        F.interpolate(tensors[i], (fix_size, fix_size), antialias=True, mode='bilinear')
                    for i in range(batch_size)], dim=0)
            elif example_shape[0] == 3 and example_shape[0] < min(example_shape[2:]):
                # B*[C, T, H, W]
                if fix_depth is not None and example_shape[1] > 3:
                    results = []
                    for i in range(batch_size):
                        t = rearrange(tensors[i], 'c t h w -> t c h w')
                        t = F.interpolate(t, (fix_size, fix_size), antialias=True, mode='bilinear')
                        t = rearrange(t, 't c h w -> (h w) c t')
                        t = F.interpolate(t, size=fix_depth, mode='linear', align_corners=False)
                        results.append(rearrange(t, '(h w) c t -> t c h w', h=fix_size, w=fix_size))
                    tensors = torch.cat(results, dim=0)
                else:
                    tensors = torch.cat([
                        F.interpolate(tensors[i].permute(1, 0, 2, 3), (fix_size, fix_size), antialias=True, mode='bilinear')
                    for i in range(batch_size)], dim=0)
            else:
                raise ValueError(f"Invalid shape for a input tensor list with 4D tensors, expected B*[3, T, H, W], got {tensor_shapes}")
        else:
            raise ValueError(f"Invalid tensor shapes in input tensor list. Expected [C, H, W] or [B, C, H, W], got: {tensor_shapes}")
    else:
        raise ValueError(f"Invalid input type. Expected torch.Tensor or list/tuple of torch.Tensor, got {type(tensors)}")

    return tensors



class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, lpips_path, use_dropout=False):    # do not use dropout by default because we use .eval mode by default
        super().__init__()
        # build models
        self.net = Vgg16(requires_grad=False)
        self.lins = nn.ModuleList([NetLinLayer(c, use_dropout=use_dropout) for c in [64, 128, 256, 512, 512]])  # c: vgg16 feature dimensions
        
        # detach parameters & set to eval mode
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        
        # load weights
        self.load_state_dict(torch.load(lpips_path, map_location='cpu'), strict=True)
        
        # register helper tensors
        self.register_buffer('shift', torch.tensor([-.030, -.088, -.188], dtype=torch.float32).view(1, 3, 1, 1).contiguous())
        self.register_buffer('scale_inv', 1. / torch.tensor([.458, .448, .450], dtype=torch.float32).view(1, 3, 1, 1).contiguous())
    
    def forward(self, inp, rec):
        lpips, _ = self.perceptual_and_gram(inp, rec, compute_gram=False)
        return lpips

    @staticmethod
    def _gram_matrix(features):
        batch, channels, height, width = features.shape
        flattened = features.reshape(batch, channels, height * width)
        return torch.bmm(flattened, flattened.transpose(1, 2)).div(
            channels * height * width
        )

    def perceptual_and_gram(self, inp, rec, compute_gram=True):
        """Compute LPIPS and the mean multi-layer normalized Gram-MSE."""
        inp = tensor_list_to_batch(inp, 256)
        rec = tensor_list_to_batch(rec, 256)

        batch_size = inp.shape[0]
        joined = torch.cat((inp, rec), dim=0).sub(self.shift).mul(self.scale_inv)
        feature_pairs = self.net(joined)
        lpips = inp.new_tensor(0.0)
        gram = inp.new_tensor(0.0)
        for features, lin in zip(feature_pairs, self.lins):
            inp_features = features[:batch_size]
            rec_features = features[batch_size:]
            lpips = lpips + lin.model(
                (normalize_tensor(inp_features) - normalize_tensor(rec_features)).square()
            ).mean()
            if compute_gram:
                gram = gram + F.mse_loss(
                    self._gram_matrix(inp_features),
                    self._gram_matrix(rec_features),
                )
        return lpips, gram / len(feature_pairs)


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else [nn.Identity()]
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16().features
        self.slice1 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(4)])
        self.slice2 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(4, 9)])
        self.slice3 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(9, 16)])
        self.slice4 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(16, 23)])
        self.slice5 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(23, 30)])
        self.N_slices = 5
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        h_relu5_3 = self.slice5(h_relu4_3)
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sum(x.square(), dim=1, keepdim=True).add_(1e-9).sqrt_()
    return x / (norm_factor + eps)


