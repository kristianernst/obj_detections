# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict
import math

import torch


def giou_loss(
  boxes1: torch.Tensor,
  boxes2: torch.Tensor,
  reduction: str = "none",
  eps: float = 1e-7,
) -> torch.Tensor:
  """
  Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
  https://arxiv.org/abs/1902.09630

  Gradient-friendly IoU loss with an additional penalty that is non-zero when the
  boxes do not overlap and scales with the size of their smallest enclosing box.
  This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

  Args:
      boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
      reduction: 'none' | 'mean' | 'sum'
               'none': No reduction will be applied to the output.
               'mean': The output will be averaged.
               'sum': The output will be summed.
      eps (float): small number to prevent division by zero
  """

  x1, y1, x2, y2 = boxes1.unbind(dim=-1)
  x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

  assert (x2 >= x1).all(), "bad box: x1 larger than x2"
  assert (y2 >= y1).all(), "bad box: y1 larger than y2"

  # Intersection keypoints
  xkis1 = torch.max(x1, x1g)
  ykis1 = torch.max(y1, y1g)
  xkis2 = torch.min(x2, x2g)
  ykis2 = torch.min(y2, y2g)

  intsctk = torch.zeros_like(x1)
  mask = (ykis2 > ykis1) & (xkis2 > xkis1)
  intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
  unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
  iouk = intsctk / (unionk + eps)

  # smallest enclosing box
  xc1 = torch.min(x1, x1g)
  yc1 = torch.min(y1, y1g)
  xc2 = torch.max(x2, x2g)
  yc2 = torch.max(y2, y2g)

  area_c = (xc2 - xc1) * (yc2 - yc1)
  miouk = iouk - ((area_c - unionk) / (area_c + eps))

  loss = 1 - miouk

  if reduction == "mean":
    loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
  elif reduction == "sum":
    loss = loss.sum()

  return loss


def smooth_l1_loss(input: torch.Tensor, target: torch.Tensor, beta: float, reduction: str = "none") -> torch.Tensor:
  """
  Smooth L1 loss defined in the Fast R-CNN paper as:
  ::
                    | 0.5 * x ** 2 / beta   if abs(x) < beta
      smoothl1(x) = |
                    | abs(x) - 0.5 * beta   otherwise,

  where x = input - target.

  Smooth L1 loss is related to Huber loss, which is defined as:
  ::
                  | 0.5 * x ** 2                  if abs(x) < beta
       huber(x) = |
                  | beta * (abs(x) - 0.5 * beta)  otherwise

  Smooth L1 loss is equal to huber(x) / beta. This leads to the following
  differences:

   - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
     converges to a constant 0 loss.
   - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
     converges to L2 loss.
   - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
     slope of 1. For Huber loss, the slope of the L1 segment is beta.

  Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
  portion replaced with a quadratic function such that at abs(x) = beta, its
  slope is 1. The quadratic segment smooths the L1 loss near x = 0.

  Args:
      input (Tensor): input tensor of any shape
      target (Tensor): target value tensor with the same shape as input
      beta (float): L1 to L2 change point.
          For beta values < 1e-5, L1 loss is computed.
      reduction: 'none' | 'mean' | 'sum'
               'none': No reduction will be applied to the output.
               'mean': The output will be averaged.
               'sum': The output will be summed.

  Returns:
      The loss with the reduction option applied.

  Note:
      PyTorch's builtin "Smooth L1 loss" implementation does not actually
      implement Smooth L1 loss, nor does it implement Huber loss. It implements
      the special case of both in which they are equal (beta=1).
      See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
  """
  if beta < 1e-5:
    # if beta == 0, then torch.where will result in nan gradients when
    # the chain rule is applied due to pytorch implementation details
    # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
    # zeros, rather than "no gradient"). To avoid this issue, we define
    # small values of beta to be exactly l1 loss.
    loss = torch.abs(input - target)
  else:
    n = torch.abs(input - target)
    cond = n < beta
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

  if reduction == "mean":
    loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
  elif reduction == "sum":
    loss = loss.sum()
  return loss


def diou_loss(
  boxes1: torch.Tensor,
  boxes2: torch.Tensor,
  reduction: str = "none",
  eps: float = 1e-7,
) -> torch.Tensor:
  """
  Distance Intersection over Union Loss (Zhaohui Zheng et. al)
  https://arxiv.org/abs/1911.08287
  Args:
      boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
      reduction: 'none' | 'mean' | 'sum'
               'none': No reduction will be applied to the output.
               'mean': The output will be averaged.
               'sum': The output will be summed.
      eps (float): small number to prevent division by zero
  """

  x1, y1, x2, y2 = boxes1.unbind(dim=-1)
  x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

  # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
  assert (x2 >= x1).all(), "bad box: x1 larger than x2"
  assert (y2 >= y1).all(), "bad box: y1 larger than y2"

  # Intersection keypoints
  xkis1 = torch.max(x1, x1g)
  ykis1 = torch.max(y1, y1g)
  xkis2 = torch.min(x2, x2g)
  ykis2 = torch.min(y2, y2g)

  intsct = torch.zeros_like(x1)
  mask = (ykis2 > ykis1) & (xkis2 > xkis1)
  intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
  union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
  iou = intsct / union

  # smallest enclosing box
  xc1 = torch.min(x1, x1g)
  yc1 = torch.min(y1, y1g)
  xc2 = torch.max(x2, x2g)
  yc2 = torch.max(y2, y2g)
  diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

  # centers of boxes
  x_p = (x2 + x1) / 2
  y_p = (y2 + y1) / 2
  x_g = (x1g + x2g) / 2
  y_g = (y1g + y2g) / 2
  distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

  # Eqn. (7)
  loss = 1 - iou + (distance / diag_len)
  if reduction == "mean":
    loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
  elif reduction == "sum":
    loss = loss.sum()

  return loss


# taken from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/losses.py#L66
def ciou_loss(
  boxes1: torch.Tensor,
  boxes2: torch.Tensor,
  reduction: str = "none",
  eps: float = 1e-7,
) -> torch.Tensor:
  """
  Complete Intersection over Union Loss (Zhaohui Zheng et. al)
  https://arxiv.org/abs/1911.08287
  Args:
      boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
      reduction: 'none' | 'mean' | 'sum'
               'none': No reduction will be applied to the output.
               'mean': The output will be averaged.
               'sum': The output will be summed.
      eps (float): small number to prevent division by zero
  """

  x1, y1, x2, y2 = boxes1.unbind(dim=-1)
  x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

  # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
  assert (x2 >= x1).all(), "bad box: x1 larger than x2"
  assert (y2 >= y1).all(), "bad box: y1 larger than y2"

  # Intersection keypoints
  xkis1 = torch.max(x1, x1g)
  ykis1 = torch.max(y1, y1g)
  xkis2 = torch.min(x2, x2g)
  ykis2 = torch.min(y2, y2g)

  intsct = torch.zeros_like(x1)
  mask = (ykis2 > ykis1) & (xkis2 > xkis1)
  intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
  union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
  iou = intsct / union

  # smallest enclosing box
  xc1 = torch.min(x1, x1g)
  yc1 = torch.min(y1, y1g)
  xc2 = torch.max(x2, x2g)
  yc2 = torch.max(y2, y2g)
  diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

  # centers of boxes
  x_p = (x2 + x1) / 2
  y_p = (y2 + y1) / 2
  x_g = (x1g + x2g) / 2
  y_g = (y1g + y2g) / 2
  distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

  # width and height of boxes
  w_pred = x2 - x1
  h_pred = y2 - y1
  w_gt = x2g - x1g
  h_gt = y2g - y1g
  v = (4 / (math.pi**2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
  with torch.no_grad():
    alpha = v / (1 - iou + v + eps)

  # Eqn. (10)
  loss = 1 - iou + (distance / diag_len) + alpha * v
  if reduction == "mean":
    loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
  elif reduction == "sum":
    loss = loss.sum()

  return loss
