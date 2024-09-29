import functools
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from fvcore.nn.distributed import differentiable_all_reduce
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torchvision.ops import boxes as box_ops
from torchvision.ops import roi_align

from util.dist import get_world_size

# from torch.nn import BatchNorm2d


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
BatchNorm2d = torch.nn.BatchNorm2d


def disable_torch_compiler(func):
  if TORCH_VERSION >= (2, 1):
    # Use the torch.compiler.disable decorator if supported
    @torch.compiler.disable
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      return func(*args, **kwargs)

    return wrapper
  else:
    # Return the function unchanged if torch.compiler.disable is not supported
    return func


class ROIAlign(nn.Module):
  def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
    """
    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each output
            sample. 0 to take samples densely.
        aligned (bool): if False, use the legacy implementation in
            Detectron. If True, align the results more perfectly.

    Note:
        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel indices (in our
        pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
        c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
        from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
        roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
        pixel indices and therefore it uses pixels with a slightly incorrect alignment
        (relative to our pixel model) when performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors; see
        detectron2/tests/test_roi_align.py for verification.

        The difference does not make a difference to the model's performance if
        ROIAlign is used together with conv layers.
    """
    super().__init__()
    self.output_size = output_size
    self.spatial_scale = spatial_scale
    self.sampling_ratio = sampling_ratio
    self.aligned = aligned

    from torchvision import __version__

    version = tuple(int(x) for x in __version__.split(".")[:2])
    # https://github.com/pytorch/vision/pull/2438
    assert version >= (0, 7), "Require torchvision >= 0.7"

  def forward(self, input, rois):
    """
    Args:
        input: NCHW images
        rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
    """
    assert rois.dim() == 2 and rois.size(1) == 5
    if input.is_quantized:
      input = input.dequantize()
    return roi_align(
      input,
      rois.to(dtype=input.dtype),
      self.output_size,
      self.spatial_scale,
      self.sampling_ratio,
      self.aligned,
    )

  def __repr__(self):
    tmpstr = self.__class__.__name__ + "("
    tmpstr += "output_size=" + str(self.output_size)
    tmpstr += ", spatial_scale=" + str(self.spatial_scale)
    tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
    tmpstr += ", aligned=" + str(self.aligned)
    tmpstr += ")"
    return tmpstr


class _ROIAlignRotated(Function):
  @staticmethod
  @disable_torch_compiler
  def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
    ctx.save_for_backward(roi)
    ctx.output_size = _pair(output_size)
    ctx.spatial_scale = spatial_scale
    ctx.sampling_ratio = sampling_ratio
    ctx.input_shape = input.size()
    output = torch.ops.detectron2.roi_align_rotated_forward(input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio)
    return output

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_output):
    (rois,) = ctx.saved_tensors
    output_size = ctx.output_size
    spatial_scale = ctx.spatial_scale
    sampling_ratio = ctx.sampling_ratio
    bs, ch, h, w = ctx.input_shape
    grad_input = torch.ops.detectron2.roi_align_rotated_backward(
      grad_output,
      rois,
      spatial_scale,
      output_size[0],
      output_size[1],
      bs,
      ch,
      h,
      w,
      sampling_ratio,
    )
    return grad_input, None, None, None, None, None


roi_align_rotated = _ROIAlignRotated.apply


class ROIAlignRotated(nn.Module):
  def __init__(self, output_size, spatial_scale, sampling_ratio):
    """
    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each output
            sample. 0 to take samples densely.

    Note:
        ROIAlignRotated supports continuous coordinate by default:
        Given a continuous coordinate c, its two neighboring pixel indices (in our
        pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
        c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
        from the underlying signal at continuous coordinates 0.5 and 1.5).
    """
    super(ROIAlignRotated, self).__init__()
    self.output_size = output_size
    self.spatial_scale = spatial_scale
    self.sampling_ratio = sampling_ratio

  def forward(self, input, rois):
    """
    Args:
        input: NCHW images
        rois: Bx6 boxes. First column is the index into N.
            The other 5 columns are (x_ctr, y_ctr, width, height, angle_degrees).
    """
    assert rois.dim() == 2 and rois.size(1) == 6
    orig_dtype = input.dtype
    if orig_dtype == torch.float16:
      input = input.float()
      rois = rois.float()
    output_size = _pair(self.output_size)

    # Scripting for Autograd is currently unsupported.
    # This is a quick fix without having to rewrite code on the C++ side
    if torch.jit.is_scripting() or torch.jit.is_tracing():
      return torch.ops.detectron2.roi_align_rotated_forward(
        input, rois, self.spatial_scale, output_size[0], output_size[1], self.sampling_ratio
      ).to(dtype=orig_dtype)

    return roi_align_rotated(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio).to(dtype=orig_dtype)

  def __repr__(self):
    tmpstr = self.__class__.__name__ + "("
    tmpstr += "output_size=" + str(self.output_size)
    tmpstr += ", spatial_scale=" + str(self.spatial_scale)
    tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
    tmpstr += ")"
    return tmpstr


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float):
  """
  Same as torchvision.ops.boxes.batched_nms, but with float().
  """
  assert boxes.shape[-1] == 4
  # Note: Torchvision already has a strategy (https://github.com/pytorch/vision/issues/1311)
  # to decide whether to use coordinate trick or for loop to implement batched_nms. So we
  # just call it directly.
  # Fp16 does not have enough range for batched NMS, so adding float().
  return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)


def nonzero_tuple(x):
  """
  A 'as_tuple=True' version of torch.nonzero to support torchscript.
  because of https://github.com/pytorch/pytorch/issues/38718
  """
  if torch.jit.is_scripting():
    if x.dim() == 0:
      return x.unsqueeze(0).nonzero().unbind(1)
    return x.nonzero().unbind(1)
  else:
    return x.nonzero(as_tuple=True)


def cat(tensors: List[torch.Tensor], dim: int = 0):
  """
  Efficient version of torch.cat that avoids a copy if there is only a single element in a list
  """
  assert isinstance(tensors, (list, tuple))
  if len(tensors) == 1:
    return tensors[0]
  return torch.cat(tensors, dim)


@torch.jit.script_if_tracing
def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
  """
  Tracing friendly way to cast tensor to another tensor's device. Device will be treated
  as constant during tracing, scripting the casting process as whole can workaround this issue.
  """
  return src.to(dst.device)


def shapes_to_tensor(x: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
  """
  Turn a list of integer scalars or integer Tensor scalars into a vector,
  in a way that's both traceable and scriptable.

  In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
  In scripting or eager, `x` should be a list of int.
  """
  if torch.jit.is_scripting():
    return torch.as_tensor(x, device=device)
  if torch.jit.is_tracing():
    assert all([isinstance(t, torch.Tensor) for t in x]), "Shape should be tensor during tracing!"
    # as_tensor should not be used in tracing because it records a constant
    ret = torch.stack(x)
    if ret.device != device:  # avoid recording a hard-coded device if not necessary
      ret = ret.to(device=device)
    return ret
  return torch.as_tensor(x, device=device)


def subsample_labels(labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int):
  """
  Return `num_samples` (or fewer, if not enough found)
  random samples from `labels` which is a mixture of positives & negatives.
  It will try to return as many positives as possible without
  exceeding `positive_fraction * num_samples`, and then try to
  fill the remaining slots with negatives.

  Args:
      labels (Tensor): (N, ) label vector with values:
          * -1: ignore
          * bg_label: background ("negative") class
          * otherwise: one or more foreground ("positive") classes
      num_samples (int): The total number of labels with value >= 0 to return.
          Values that are not sampled will be filled with -1 (ignore).
      positive_fraction (float): The number of subsampled labels with values > 0
          is `min(num_positives, int(positive_fraction * num_samples))`. The number
          of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
          In order words, if there are not enough positives, the sample is filled with
          negatives. If there are also not enough negatives, then as many elements are
          sampled as is possible.
      bg_label (int): label index of background ("negative") class.

  Returns:
      pos_idx, neg_idx (Tensor):
          1D vector of indices. The total length of both is `num_samples` or fewer.
  """
  positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
  negative = nonzero_tuple(labels == bg_label)[0]

  num_pos = int(num_samples * positive_fraction)
  # protect against not enough positive examples
  num_pos = min(positive.numel(), num_pos)
  num_neg = num_samples - num_pos
  # protect against not enough negative examples
  num_neg = min(negative.numel(), num_neg)

  # randomly select positive and negative examples
  perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
  perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

  pos_idx = positive[perm1]
  neg_idx = negative[perm2]
  return pos_idx, neg_idx


class CNNBlockBase(nn.Module):
  """
  A CNN block is assumed to have input channels, output channels and a stride.
  The input and output of `forward()` method must be NCHW tensors.
  The method can perform arbitrary computation but must match the given
  channels and stride specification.

  Attribute:
      in_channels (int):
      out_channels (int):
      stride (int):
  """

  def __init__(self, in_channels, out_channels, stride):
    """
    The `__init__` method of any subclass should also contain these arguments.

    Args:
        in_channels (int):
        out_channels (int):
        stride (int):
    """
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride

  def freeze(self):
    """
    Make this block not trainable.
    This method sets all parameters to `requires_grad=False`,
    and convert all BatchNorm layers to FrozenBatchNorm

    Returns:
        the block itself
    """
    for p in self.parameters():
      p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(self)
    return self


class FrozenBatchNorm2d(nn.Module):
  """
  BatchNorm2d where the batch statistics and the affine parameters are fixed.

  It contains non-trainable buffers called
  "weight" and "bias", "running_mean", "running_var",
  initialized to perform identity transformation.

  The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
  which are computed from the original four parameters of BN.
  The affine transform `x * weight + bias` will perform the equivalent
  computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
  When loading a backbone model from Caffe2, "running_mean" and "running_var"
  will be left unchanged as identity transformation.

  Other pre-trained backbone models may contain all 4 parameters.

  The forward is implemented by `F.batch_norm(..., training=False)`.
  """

  _version = 3

  def __init__(self, num_features, eps=1e-5):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.register_buffer("weight", torch.ones(num_features))
    self.register_buffer("bias", torch.zeros(num_features))
    self.register_buffer("running_mean", torch.zeros(num_features))
    self.register_buffer("running_var", torch.ones(num_features) - eps)
    self.register_buffer("num_batches_tracked", None)

  def forward(self, x):
    if x.requires_grad:
      # When gradients are needed, F.batch_norm will use extra memory
      # because its backward op computes gradients for weight/bias as well.
      scale = self.weight * (self.running_var + self.eps).rsqrt()
      bias = self.bias - self.running_mean * scale
      scale = scale.reshape(1, -1, 1, 1)
      bias = bias.reshape(1, -1, 1, 1)
      out_dtype = x.dtype  # may be half
      return x * scale.to(out_dtype) + bias.to(out_dtype)
    else:
      # When gradients are not needed, F.batch_norm is a single fused op
      # and provide more optimization opportunities.
      return F.batch_norm(
        x,
        self.running_mean,
        self.running_var,
        self.weight,
        self.bias,
        training=False,
        eps=self.eps,
      )

  def _load_from_state_dict(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
  ):
    version = local_metadata.get("version", None)

    if version is None or version < 2:
      # No running_mean/var in early versions
      # This will silent the warnings
      if prefix + "running_mean" not in state_dict:
        state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
      if prefix + "running_var" not in state_dict:
        state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

    super()._load_from_state_dict(
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
    )

  def __repr__(self):
    return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

  @classmethod
  def convert_frozen_batchnorm(cls, module):
    """
    Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

    Args:
        module (torch.nn.Module):

    Returns:
        If module is BatchNorm/SyncBatchNorm, returns a new module.
        Otherwise, in-place convert module and return it.

    Similar to convert_sync_batchnorm in
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
    """
    bn_module = nn.modules.batchnorm
    bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
    res = module
    if isinstance(module, bn_module):
      res = cls(module.num_features)
      if module.affine:
        res.weight.data = module.weight.data.clone().detach()
        res.bias.data = module.bias.data.clone().detach()
      res.running_mean.data = module.running_mean.data
      res.running_var.data = module.running_var.data
      res.eps = module.eps
      res.num_batches_tracked = module.num_batches_tracked
    else:
      for name, child in module.named_children():
        new_child = cls.convert_frozen_batchnorm(child)
        if new_child is not child:
          res.add_module(name, new_child)
    return res

  @classmethod
  def convert_frozenbatchnorm2d_to_batchnorm2d(cls, module: nn.Module) -> nn.Module:
    """
    Convert all FrozenBatchNorm2d to BatchNorm2d

    Args:
        module (torch.nn.Module):

    Returns:
        If module is FrozenBatchNorm2d, returns a new module.
        Otherwise, in-place convert module and return it.

    This is needed for quantization:
        https://fb.workplace.com/groups/1043663463248667/permalink/1296330057982005/
    """

    res = module
    if isinstance(module, FrozenBatchNorm2d):
      res = torch.nn.BatchNorm2d(module.num_features, module.eps)

      res.weight.data = module.weight.data.clone().detach()
      res.bias.data = module.bias.data.clone().detach()
      res.running_mean.data = module.running_mean.data.clone().detach()
      res.running_var.data = module.running_var.data.clone().detach()
      res.eps = module.eps
      res.num_batches_tracked = module.num_batches_tracked
    else:
      for name, child in module.named_children():
        new_child = cls.convert_frozenbatchnorm2d_to_batchnorm2d(child)
        if new_child is not child:
          res.add_module(name, new_child)
    return res


def get_norm(norm, out_channels):
  """
  Args:
      norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
          or a callable that takes a channel number and returns
          the normalization layer as a nn.Module.

  Returns:
      nn.Module or None: the normalization layer
  """
  if norm is None:
    return None
  if isinstance(norm, str):
    if len(norm) == 0:
      return None
    norm = {
      "BN": BatchNorm2d,
      # Fixed in https://github.com/pytorch/pytorch/pull/36382
      "SyncBN": NaiveSyncBatchNorm if TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
      "FrozenBN": FrozenBatchNorm2d,
      "GN": lambda channels: nn.GroupNorm(32, channels),
      # for debugging:
      "nnSyncBN": nn.SyncBatchNorm,
      "naiveSyncBN": NaiveSyncBatchNorm,
      # expose stats_mode N as an option to caller, required for zero-len inputs
      "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
      "LN": lambda channels: LayerNorm(channels),
    }[norm]
  return norm(out_channels)


class NaiveSyncBatchNorm(BatchNorm2d):
  """
  In PyTorch<=1.5, ``nn.SyncBatchNorm`` has incorrect gradient
  when the batch size on each worker is different.
  (e.g., when scale augmentation is used, or when it is applied to mask head).

  This is a slower but correct alternative to `nn.SyncBatchNorm`.

  Note:
      There isn't a single definition of Sync BatchNorm.

      When ``stats_mode==""``, this module computes overall statistics by using
      statistics of each worker with equal weight.  The result is true statistics
      of all samples (as if they are all on one worker) only when all workers
      have the same (N, H, W). This mode does not support inputs with zero batch size.

      When ``stats_mode=="N"``, this module computes overall statistics by weighting
      the statistics of each worker by their ``N``. The result is true statistics
      of all samples (as if they are all on one worker) only when all workers
      have the same (H, W). It is slower than ``stats_mode==""``.

      Even though the result of this module may not be the true statistics of all samples,
      it may still be reasonable because it might be preferrable to assign equal weights
      to all workers, regardless of their (H, W) dimension, instead of putting larger weight
      on larger images. From preliminary experiments, little difference is found between such
      a simplified implementation and an accurate computation of overall mean & variance.
  """

  def __init__(self, *args, stats_mode="", **kwargs):
    super().__init__(*args, **kwargs)
    assert stats_mode in ["", "N"]
    self._stats_mode = stats_mode

  def forward(self, input):
    if get_world_size() == 1 or not self.training:
      return super().forward(input)

    B, C = input.shape[0], input.shape[1]

    half_input = input.dtype == torch.float16
    if half_input:
      # fp16 does not have good enough numerics for the reduction here
      input = input.float()
    mean = torch.mean(input, dim=[0, 2, 3])
    meansqr = torch.mean(input * input, dim=[0, 2, 3])

    if self._stats_mode == "":
      assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
      vec = torch.cat([mean, meansqr], dim=0)
      vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
      mean, meansqr = torch.split(vec, C)
      momentum = self.momentum
    else:
      if B == 0:
        vec = torch.zeros([2 * C + 1], device=mean.device, dtype=mean.dtype)
        vec = vec + input.sum()  # make sure there is gradient w.r.t input
      else:
        vec = torch.cat(
          [
            mean,
            meansqr,
            torch.ones([1], device=mean.device, dtype=mean.dtype),
          ],
          dim=0,
        )
      vec = differentiable_all_reduce(vec * B)

      total_batch = vec[-1].detach()
      momentum = total_batch.clamp(max=1) * self.momentum  # no update if total_batch is 0
      mean, meansqr, _ = torch.split(vec / total_batch.clamp(min=1), C)  # avoid div-by-zero

    var = meansqr - mean * mean
    invstd = torch.rsqrt(var + self.eps)
    scale = self.weight * invstd
    bias = self.bias - mean * scale
    scale = scale.reshape(1, -1, 1, 1)
    bias = bias.reshape(1, -1, 1, 1)

    self.running_mean += momentum * (mean.detach() - self.running_mean)
    self.running_var += momentum * (var.detach() - self.running_var)
    ret = input * scale + bias
    if half_input:
      ret = ret.half()
    return ret


class LayerNorm(nn.Module):
  """
  A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
  variance normalization over the channel dimension for inputs that have shape
  (batch_size, channels, height, width).
  https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
  """

  def __init__(self, normalized_shape, eps=1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(normalized_shape))
    self.bias = nn.Parameter(torch.zeros(normalized_shape))
    self.eps = eps
    self.normalized_shape = (normalized_shape,)

  def forward(self, x):
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + self.eps)
    x = self.weight[:, None, None] * x + self.bias[:, None, None]
    return x


def comm_get_world_size() -> int:
  if not dist.is_available():
    return 1
  if not dist.is_initialized():
    return 1
  return dist.get_world_size()


@dataclass
class ShapeSpec:
  """
  A simple structure that contains basic shape specification about a tensor.
  It is often used as the auxiliary inputs/outputs of models,
  to complement the lack of shape inference ability among pytorch modules.
  """

  channels: Optional[int] = None
  height: Optional[int] = None
  width: Optional[int] = None
  stride: Optional[int] = None


class LastLevelMaxPool(nn.Module):
  """
  This module is used in the original FPN to generate a downsampled
  P6 feature from P5.
  """

  def __init__(self):
    super().__init__()
    self.num_levels = 1
    self.in_feature = "p5"

  def forward(self, x):
    return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


def check_if_dynamo_compiling():
  if TORCH_VERSION >= (2, 1):
    from torch._dynamo import is_compiling

    return is_compiling()
  else:
    return False


def empty_input_loss_func_wrapper(loss_func):
  def wrapped_loss_func(input, target, *, reduction="mean", **kwargs):
    """
    Same as `loss_func`, but returns 0 (instead of nan) for empty inputs.
    """
    if target.numel() == 0 and reduction == "mean":
      return input.sum() * 0.0  # connect the gradient
    return loss_func(input, target, reduction=reduction, **kwargs)

  return wrapped_loss_func


cross_entropy = empty_input_loss_func_wrapper(F.cross_entropy)

ConvTranspose2d = torch.nn.ConvTranspose2d


class Conv2d(torch.nn.Conv2d):
  """
  A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
  """

  def __init__(self, *args, **kwargs):
    """
    Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

    Args:
        norm (nn.Module, optional): a normalization layer
        activation (callable(Tensor) -> Tensor): a callable activation function

    It assumes that norm layer is used before activation.
    """
    norm = kwargs.pop("norm", None)
    activation = kwargs.pop("activation", None)
    super().__init__(*args, **kwargs)

    self.norm = norm
    self.activation = activation

  def forward(self, x):
    # torchscript does not support SyncBatchNorm yet
    # https://github.com/pytorch/pytorch/issues/40507
    # and we skip these codes in torchscript since:
    # 1. currently we only support torchscript in evaluation mode
    # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
    # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
    if not torch.jit.is_scripting():
      # Dynamo doesn't support context managers yet
      is_dynamo_compiling = check_if_dynamo_compiling()
      if not is_dynamo_compiling:
        with warnings.catch_warnings(record=True):
          if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(self.norm, torch.nn.SyncBatchNorm), "SyncBatchNorm does not support empty inputs!"

    x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    if self.norm is not None:
      x = self.norm(x)
    if self.activation is not None:
      x = self.activation(x)
    return x


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
  """
  Args:
      masks: N, 1, H, W
      boxes: N, 4
      img_h, img_w (int):
      skip_empty (bool): only paste masks within the region that
          tightly bound all boxes, and returns the results this region only.
          An important optimization for CPU.

  Returns:
      if skip_empty == False, a mask of shape (N, img_h, img_w)
      if skip_empty == True, a mask of shape (N, h', w'), and the slice
          object for the corresponding region.
  """
  # On GPU, paste all masks together (up to chunk size)
  # by using the entire image to sample the masks
  # Compared to pasting them one by one,
  # this has more operations but is faster on COCO-scale dataset.
  device = masks.device

  if skip_empty and not torch.jit.is_scripting():
    x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(dtype=torch.int32)
    x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
    y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
  else:
    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
  x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

  N = masks.shape[0]

  img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
  img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
  img_y = (img_y - y0) / (y1 - y0) * 2 - 1
  img_x = (img_x - x0) / (x1 - x0) * 2 - 1
  # img_x, img_y have shapes (N, w), (N, h)

  gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
  gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
  grid = torch.stack([gx, gy], dim=3)

  if not torch.jit.is_scripting():
    if not masks.dtype.is_floating_point:
      masks = masks.float()
  img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

  if skip_empty and not torch.jit.is_scripting():
    return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
  else:
    return img_masks[:, 0], ()


@torch.jit.script_if_tracing
def paste_masks_in_image(masks: torch.Tensor, boxes: torch.Tensor, image_shape: Tuple[int, int], threshold: float = 0.5):
  """
  Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
  The location, height, and width for pasting each mask is determined by their
  corresponding bounding boxes in boxes.

  Note:
      This is a complicated but more accurate implementation. In actual deployment, it is
      often enough to use a faster but less accurate implementation.
      See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

  Args:
      masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
          detected object instances in the image and Hmask, Wmask are the mask width and mask
          height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
      boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
          boxes[i] and masks[i] correspond to the same object instance.
      image_shape (tuple): height, width
      threshold (float): A threshold in [0, 1] for converting the (soft) masks to
          binary masks.

  Returns:
      img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
      number of detected object instances and Himage, Wimage are the image width
      and height. img_masks[i] is a binary mask for object instance i.
  """

  assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
  N = len(masks)
  if N == 0:
    return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
  if not isinstance(boxes, torch.Tensor):
    boxes = boxes.tensor
  device = boxes.device
  assert len(boxes) == N, boxes.shape

  img_h, img_w = image_shape

  # The actual implementation split the input into chunks,
  # and paste them chunk by chunk.
  if device.type == "cpu" or torch.jit.is_scripting():
    # CPU is most efficient when they are pasted one by one with skip_empty=True
    # so that it performs minimal number of operations.
    num_chunks = N
  else:
    # GPU benefits from parallelism for larger chunks, but may have memory issue
    # int(img_h) because shape may be tensors in tracing
    num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
    assert num_chunks <= N, "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
  chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

  img_masks = torch.zeros(N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8)
  for inds in chunks:
    masks_chunk, spatial_inds = _do_paste_mask(masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu")

    if threshold >= 0:
      masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
    else:
      # for visualization and debugging
      masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

    if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
      img_masks[inds] = masks_chunk
    else:
      img_masks[(inds,) + spatial_inds] = masks_chunk
  return img_masks


@torch.jit.script_if_tracing
def _paste_masks_tensor_shape(
  masks: torch.Tensor,
  boxes: torch.Tensor,
  image_shape: Tuple[torch.Tensor, torch.Tensor],
  threshold: float = 0.5,
):
  """
  A wrapper of paste_masks_in_image where image_shape is Tensor.
  During tracing, shapes might be tensors instead of ints. The Tensor->int
  conversion should be scripted rather than traced.
  """
  return paste_masks_in_image(masks, boxes, (int(image_shape[0]), int(image_shape[1])), threshold)


# def check_if_dynamo_compiling():
#     if TORCH_VERSION >= (2, 1):
#         from torch._dynamo import is_compiling

#         return is_compiling()
#     else:
#         return False

# class Conv2d(torch.nn.Conv2d):
#     """
#     A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
#     """

#     def __init__(self, *args, **kwargs):
#         """
#         Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

#         Args:
#             norm (nn.Module, optional): a normalization layer
#             activation (callable(Tensor) -> Tensor): a callable activation function

#         It assumes that norm layer is used before activation.
#         """
#         norm = kwargs.pop("norm", None)
#         activation = kwargs.pop("activation", None)
#         super().__init__(*args, **kwargs)

#         self.norm = norm
#         self.activation = activation

#     def forward(self, x):
#         # torchscript does not support SyncBatchNorm yet
#         # https://github.com/pytorch/pytorch/issues/40507
#         # and we skip these codes in torchscript since:
#         # 1. currently we only support torchscript in evaluation mode
#         # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
#         # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
#         if not torch.jit.is_scripting():
#             # Dynamo doesn't support context managers yet
#             is_dynamo_compiling = check_if_dynamo_compiling()
#             if not is_dynamo_compiling:
#                 with warnings.catch_warnings(record=True):
#                     if x.numel() == 0 and self.training:
#                         # https://github.com/pytorch/pytorch/issues/12013
#                         assert not isinstance(
#                             self.norm, torch.nn.SyncBatchNorm
#                         ), "SyncBatchNorm does not support empty inputs!"

#         x = F.conv2d(
#             x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
#         )
#         if self.norm is not None:
#             x = self.norm(x)
#         if self.activation is not None:
#             x = self.activation(x)
#         return x
