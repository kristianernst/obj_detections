from typing import Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

"""
Taken from: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/utils.py
"""


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    -> Basically, we take an image, make patches of it and then embed the patches - done efficiently with convolutions.

    https://github.com/lucidrains/vit-pytorch
    """
    
    def __init__(
        self, 
        kernel_size: Tuple[int, int]=(16,16), 
        stride: Tuple[int, int]=(16,16), 
        padding: Tuple[int, int]=(0,0), 
        in_chans: int=3, 
        embed_dim: int=768
    ):
        """
        kernel_size: height and width of the patch
        stride: height and width of the stride
        padding: height and width of the padding
        in_chans: number of input channels
        embed_dim: the embedding dimension
        """
        super().__init__()
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = torch.einsum("bchw -> bhwc", x)
        return x
 
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]
    
def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Add decomposed relative position bias to the attention map.
    
    Args:
        attn: attention map
        q: query with shape (B, q_h * q_w, C)
        rel_pos_h: relative position embedding for height axis (Lh, C)
        rel_pos_w: relative position embedding for width axis (Lw, C)
        q_size: spatial sequence size of query q of shape (q_h, q_w)
        k_size: spatial sequence size of key k of shape (k_h, k_w)
    """
    
    q_h, q_w = q_size
    k_h, k_w = k_size
    
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc, hkc -> bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc, wkc -> bhwk", r_q, Rw)
    
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:,:,:,:, None] + rel_w[:,:,:, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    
    return attn


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


# def get_abs_pos(abs_pos, has_cls_token, hw):
#     """
#     Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
#         dimension for the original embeddings.
#     Args:
#         abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
#         has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
#         hw (Tuple): size of input image tokens.

#     Returns:
#         Absolute positional embeddings after processing with shape (1, H, W, C)
#     """
#     h, w = hw
#     if has_cls_token:
#         abs_pos = abs_pos[:, 1:]
#     xy_num = abs_pos.shape[1]
#     size = int(math.sqrt(xy_num))
#     assert size * size == xy_num

#     if size != h or size != w:
#         new_abs_pos = F.interpolate(
#             abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
#             size=(h, w),
#             mode="bicubic",
#             align_corners=False,
#         )

#         return new_abs_pos.permute(0, 2, 3, 1)
#     else:
#         return abs_pos.reshape(1, h, w, -1)

# TODO: verify abs pos function. seemingly it works with MPS.
def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        # Reshape abs_pos to (1, C, size, size)
        abs_pos_reshaped = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        
        # Create a new tensor with target size
        new_abs_pos = torch.zeros((1, abs_pos_reshaped.shape[1], h, w), device=abs_pos.device)
        
        # Compute scaling factors
        scale_h, scale_w = h / size, w / size
        
        for i in range(h):
            for j in range(w):
                # Compute source indices
                src_i, src_j = int(i / scale_h), int(j / scale_w)
                
                # Copy values
                new_abs_pos[0, :, i, j] = abs_pos_reshaped[0, :, src_i, src_j]
        
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)
    

        
        