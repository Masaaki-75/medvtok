"""
Adapted from OpenCLIP
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


# usage: like tot_loss = hinge_loss(logits_real) + hinge_loss(-logits_fake)
def hinge_loss(logits: torch.Tensor):
    return (1 - logits).relu().mean()


def softplus_loss(logits: torch.Tensor):
    return F.softplus(-logits).mean()


def linear_loss(logits: torch.Tensor):
    return (-logits).mean()


def focal_l1_loss(
    pred, target, reduction='none',
    alpha=0.2, gamma=1.0, activate='sigmoid', residual=False, weight=None
):
    r"""Calculate Focal L1 loss.

    Delving into Deep Imbalanced Regression. In ICML, 2021.
    <https://arxiv.org/abs/2102.09554>

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        alpha (float): A balanced form for Focal Loss. Defaults to 0.2.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 1.0.
        activate (str): activate methods in Focal loss in {'sigmoid', 'tanh'}.
            Defaults to 'sigmoid'.
        residual (bool): Whether to use the original l1_loss, i.e., l1 + focal_l1.
            Defaults to False.
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    _loss = F.l1_loss(pred, target, reduction='none')
    if activate == 'tanh':
        loss = _loss * (torch.tanh(alpha * _loss)) ** gamma
    else:
        loss = _loss * (2. * torch.sigmoid(alpha * _loss) - 1.) ** gamma
    if residual:
        loss += _loss
    
    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def cosinesim_loss(feat1, feat2, dim=-1):
    cos_sim = F.cosine_similarity(feat1, feat2, dim=dim)
    loss = 1 - cos_sim
    return torch.mean(loss)


def compute_expert_alignment_loss(projected_features, expert_features, expert_ddp_anchor=None):
    """Align backprojected tokenizer features with their native expert features.

    The cosine loss includes an anchor to keep every expert projector in the
    distributed gradient graph.
    """
    if isinstance(projected_features, (list, tuple)) and isinstance(expert_features, (list, tuple)):
        batch_size = len(projected_features)

        loss = 0.0
        for i in range(batch_size):
            projected_feat = projected_features[i]  # [1, expert_dim]
            expert_feat = expert_features[i].to(device=projected_feat.device, dtype=projected_feat.dtype)

            cos_sim = F.cosine_similarity(projected_feat, expert_feat, dim=-1)
            loss = loss + (1.0 - cos_sim)
        loss = loss.sum() / batch_size

    elif isinstance(projected_features, torch.Tensor) and isinstance(expert_features, torch.Tensor):
        cos_sim = F.cosine_similarity(projected_features, expert_features, dim=-1)
        loss = (1.0 - cos_sim).mean()

    # Keep all expert-space projectors in the gradient graph.
    if expert_ddp_anchor is not None:
        loss = loss + 0 * expert_ddp_anchor
    
    return loss


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features



class ClipLoss(nn.Module):

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False, **kwargs):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):

        clip_loss = torch.tensor(0)

        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
                                   F.cross_entropy(logits_per_image, labels) +
                                   F.cross_entropy(logits_per_text, labels)
                           ) / 2

        distill_loss = (
                               self.dist_loss(dist_logits_per_image, logits_per_image) +
                               self.dist_loss(dist_logits_per_text, logits_per_text)
                       ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
               NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
        self,
        cache_labels=False,
        rank=0,
        world_size=1,
        bidir=True,
        use_horovod=False,
        *args, **kwargs,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss




def _single_volume_slidr_terms(x, y, second_order_weight):
    """
    x, y: [C, D, H, W]
    Returns:
        first_loss, second_loss, has_first, has_second
    """
    zero = y.sum() * 0.0  # keeps device/dtype and gradient graph
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.ndim != 4:
        raise ValueError(f"Expected [C, D, H, W], got shape {tuple(x.shape)}")
        
    depth = x.shape[1]

    # First-order slice differences: x[:, k+1] - x[:, k]
    if depth >= 2:
        dx = x[:, 1:, :, :] - x[:, :-1, :, :]
        dy = y[:, 1:, :, :] - y[:, :-1, :, :]
        first_loss = F.l1_loss(dy, dx, reduction="mean")
        has_first = True
    else:
        first_loss = zero
        has_first = False

    # Second-order slice differences: x[:, k+1] - 2*x[:, k] + x[:, k-1]
    if depth >= 3 and second_order_weight > 0:
        ddx = x[:, 2:, :, :] - 2.0 * x[:, 1:-1, :, :] + x[:, :-2, :, :]
        ddy = y[:, 2:, :, :] - 2.0 * y[:, 1:-1, :, :] + y[:, :-2, :, :]
        second_loss = F.l1_loss(ddy, ddx, reduction="mean")
        has_second = True
    else:
        second_loss = zero
        has_second = False

    return first_loss, second_loss, has_first, has_second


def _check_single_pair(x, y, idx=None):
    prefix = f"[sample {idx}] " if idx is not None else ""
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError(f"{prefix}Both inputs and recons must be torch.Tensor.")
    if x.shape != y.shape:
        raise ValueError(f"{prefix}Shape mismatch: inputs.shape={tuple(x.shape)} vs recons.shape={tuple(y.shape)}")


def slice_differential_regularization_loss(
    inputs,
    recons,
    second_order_weight: float = 1.0,
):
    """
    Slice Differential Regularization (SLIDR) for 3D volumes.

    Supports either:
      1) Batched tensors:
         - inputs: [B, C, D, H, W]
         - recons: [B, C, D, H, W]
      2) Lists / tuples of tensors with possibly different spatial sizes:
         - inputs[i]: [C, D, H, W]
         - recons[i]: [C, D, H, W]

    The loss is:
        mean_k || (recons[:, :, k+1] - recons[:, :, k]) - (inputs[:, :, k+1] - inputs[:, :, k]) ||_1
      + second_order_weight
        * mean_k || (recons[:, :, k+1] - 2*recons[:, :, k] + recons[:, :, k-1])
                     - (inputs[:, :, k+1] - 2*inputs[:, :, k] + inputs[:, :, k-1]) ||_1

    along the depth axis.

    Notes:
    - If depth < 2, the first-order term is skipped.
    - If depth < 3, the second-order term is skipped.
    - If all samples are too shallow for a term, that term contributes zero.
    - Returns a scalar tensor on the same device/dtype as `recons`.

    Args:
        inputs: torch.Tensor or list/tuple of torch.Tensor
        recons: torch.Tensor or list/tuple of torch.Tensor
        second_order_weight: weight for the second-order slice-difference term;
            defaults to 1.0 so SLIDR is first-order + second-order.

    Returns:
        loss: scalar torch.Tensor
    """
    # Case 1: batched tensor input
    if torch.is_tensor(inputs) and torch.is_tensor(recons):
        if inputs.shape != recons.shape:
            raise ValueError(f"Shape mismatch: inputs.shape={tuple(inputs.shape)} vs recons.shape={tuple(recons.shape)}")
        if inputs.ndim != 5:
            raise ValueError(f"For batched tensor input, expected shape [B, C, D, H, W], got {tuple(inputs.shape)}")

        zero = recons.sum() * 0.0
        first_loss_total = zero
        second_loss_total = zero
        first_count = 0
        second_count = 0

        for b in range(inputs.shape[0]):
            first_loss, second_loss, has_first, has_second = _single_volume_slidr_terms(
                inputs[b], recons[b], second_order_weight=second_order_weight
            )
            if has_first:
                first_loss_total = first_loss_total + first_loss
                first_count += 1
            if has_second:
                second_loss_total = second_loss_total + second_loss
                second_count += 1

        first_term = first_loss_total / first_count if first_count > 0 else zero
        second_term = second_loss_total / second_count if second_count > 0 else zero

        return first_term + second_order_weight * second_term

    # Case 2: list/tuple input
    elif isinstance(inputs, (list, tuple)) and isinstance(recons, (list, tuple)):
        if len(inputs) != len(recons):
            raise ValueError(f"Length mismatch: len(inputs)={len(inputs)} vs len(recons)={len(recons)}")
        if len(inputs) == 0:
            raise ValueError("inputs and recons must not be empty.")

        for i, (x, y) in enumerate(zip(inputs, recons)):
            _check_single_pair(x, y, idx=i)

        zero = recons[0].sum() * 0.0
        first_loss_total = zero
        second_loss_total = zero
        first_count = 0
        second_count = 0

        for x, y in zip(inputs, recons):
            first_loss, second_loss, has_first, has_second = _single_volume_slidr_terms(
                x, y, second_order_weight=second_order_weight
            )
            if has_first:
                first_loss_total = first_loss_total + first_loss
                first_count += 1
            if has_second:
                second_loss_total = second_loss_total + second_loss
                second_count += 1

        first_term = first_loss_total / first_count if first_count > 0 else zero
        second_term = second_loss_total / second_count if second_count > 0 else zero

        return first_term + second_order_weight * second_term

    else:
        raise TypeError(
            f"Inputs and recons must both be either:\n"
            f"  - torch.Tensor with shape [B, C, D, H, W], or\n"
            f"  - list/tuple of torch.Tensor with shape [C, D, H, W]."
            f"But got: input type {type(inputs)} and recon type {type(recons)}"
        )


def _mean_z_regularization_loss(recons, per_volume_loss, loss_name):
    """
    Apply a z-axis regularizer to a batch or a variable-shape list of volumes.

    Args:
        recons: A tensor [B, C, D, H, W], or a list/tuple whose elements are
            tensors [C, D, H, W].
        per_volume_loss: Callable returning ``(loss, is_valid)`` for one volume.
        loss_name: Name used in validation error messages.

    Returns:
        Scalar tensor averaged over volumes with sufficient depth. If no volume
        has sufficient depth, returns a graph-connected zero.
    """
    if torch.is_tensor(recons):
        if recons.ndim != 5:
            raise ValueError(
                f"{loss_name} expects [B, C, D, H, W], got {tuple(recons.shape)}"
            )
        if recons.shape[0] == 0:
            raise ValueError(f"{loss_name} received an empty batch.")
        volumes = [recons[b] for b in range(recons.shape[0])]
    elif isinstance(recons, (list, tuple)):
        if len(recons) == 0:
            raise ValueError(f"{loss_name} received an empty list/tuple.")
        volumes = list(recons)
        for i, volume in enumerate(volumes):
            if not torch.is_tensor(volume):
                raise TypeError(
                    f"{loss_name} expects tensors, but sample {i} has type {type(volume)}."
                )
            if volume.ndim != 4:
                raise ValueError(
                    f"{loss_name} expects each sample to have shape [C, D, H, W], "
                    f"but sample {i} has shape {tuple(volume.shape)}."
                )
    else:
        raise TypeError(
            f"{loss_name} expects either a torch.Tensor [B, C, D, H, W] or "
            f"a list/tuple of tensors [C, D, H, W], got {type(recons)}."
        )

    zero = volumes[0].sum() * 0.0
    total = zero
    count = 0
    for volume in volumes:
        loss, is_valid = per_volume_loss(volume)
        if is_valid:
            total = total + loss
            count += 1

    return total / count if count > 0 else zero


def total_variation_loss(recons):
    """
    Anisotropic total variation regularization along the z/depth axis.

    For each reconstructed volume ``y`` with shape [C, D, H, W]:

        mean(abs(y[:, 1:] - y[:, :-1]))

    This is a prediction-only smoothness prior: it penalizes every inter-slice
    change and does not use the ground-truth volume.

    Supports either a batched tensor [B, C, D, H, W] or a list/tuple of
    variable-shape tensors [C, D, H, W].
    """
    def _single_volume_loss(volume):
        if volume.shape[1] < 2:
            return volume.sum() * 0.0, False
        z_gradient = volume[:, 1:, :, :] - volume[:, :-1, :, :]
        return z_gradient.abs().mean(), True

    return _mean_z_regularization_loss(
        recons,
        per_volume_loss=_single_volume_loss,
        loss_name="total_variation_loss",
    )


def laplacian_loss(recons):
    """
    Laplacian regularization along the z/depth axis.

    For each reconstructed volume ``y`` with shape [C, D, H, W]:

        mean(abs(y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]))

    This is a prediction-only curvature penalty: it encourages reconstructed
    intensities to vary smoothly along z and does not use the ground truth.

    Supports either a batched tensor [B, C, D, H, W] or a list/tuple of
    variable-shape tensors [C, D, H, W].
    """
    def _single_volume_loss(volume):
        if volume.shape[1] < 3:
            return volume.sum() * 0.0, False
        z_laplacian = (
            volume[:, 2:, :, :]
            - 2.0 * volume[:, 1:-1, :, :]
            + volume[:, :-2, :, :]
        )
        return z_laplacian.abs().mean(), True

    return _mean_z_regularization_loss(
        recons,
        per_volume_loss=_single_volume_loss,
        loss_name="laplacian_loss",
    )
