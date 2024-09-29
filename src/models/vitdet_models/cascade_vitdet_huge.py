from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn

from models.backbone.vit import SimpleFeaturePyramid, ViT
from models.head.generators.anchor_generator import DefaultAnchorGenerator
from models.head.generators.box_regression import Box2BoxTransform
from models.head.generators.matcher import Matcher
from models.head.generators.proposal_generator import RPN, StandardRPNHead
from models.head.rcnn import GeneralizedRCNN
from models.head.roi.box_head import FastRCNNConvFCHead
from models.head.roi.fast_rcnn import FastRCNNOutputLayers
from models.head.roi.mask_head import MaskRCNNConvUpsampleHead
from models.head.roi.roi_heads import CascadeROIHeads
from models.layers import LastLevelMaxPool, ShapeSpec
from models.poolers import ROIPooler
from util.vars import constants

IMG_SIZE = 1024
NET_OUT_FEATURES = "last_feat"
NUM_CLASSES = 80
embed_dim, depth, num_heads, dp = 1280, 32, 16, 0.5

vit = ViT(
    img_size=IMG_SIZE,
    patch_size=16,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    drop_path_rate=dp,
    window_size=14,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    window_block_indexes=list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31)),
    residual_block_indexes=(),
    use_rel_pos=True,
    out_feature=NET_OUT_FEATURES,
)

backbone = SimpleFeaturePyramid(
    net=vit,
    in_feature=NET_OUT_FEATURES,
    out_channels=256,
    scale_factors=[4.0, 2.0, 1.0, 0.5],
    top_block=LastLevelMaxPool(),
    norm="LN",
    square_pad=IMG_SIZE,
)


standard_rpn_head = StandardRPNHead(
    in_channels=256,
    num_anchors=3,
    conv_dims=[-1, -1],  # as per vitdet
)  # conv dims is taken from https://github.com/facebookresearch/detectron2/blob/main/configs/common/models/mask_rcnn_vitdet.py

default_anchor_generator = DefaultAnchorGenerator(
    sizes=[[32], [64], [128], [256], [512]],
    aspect_ratios=[0.5, 1.0, 2.0],
    strides=[4, 8, 16, 32, 64],
    offset=0.0,
)

anchor_matcher = Matcher(thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True)

box2box_transform = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))

proposal_generator = RPN(
    in_features=["p2", "p3", "p4", "p5", "p6"],
    head=standard_rpn_head,
    anchor_generator=default_anchor_generator,
    anchor_matcher=anchor_matcher,
    box2box_transform=box2box_transform,
    batch_size_per_image=256,
    positive_fraction=0.5,
    pre_nms_topk=(2000, 1000),
    post_nms_topk=(1000, 1000),
    nms_thresh=0.7,
)

roi_pooler = ROIPooler(
    output_size=7,
    scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
    sampling_ratio=0,
    pooler_type="ROIAlignV2",
)

roi_head_matchers = [Matcher(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False) for th in [0.5, 0.6, 0.7]]

roi_box_heads = [
    FastRCNNConvFCHead(
        input_shape=ShapeSpec(channels=256, height=7, width=7), conv_dims=[256, 256, 256, 256], fc_dims=[1024], conv_norm="LN"
    )
    for _ in range(3)
]


roi_box_predictors = [
    FastRCNNOutputLayers(
        input_shape=ShapeSpec(channels=1024),
        test_score_thresh=0.05,
        box2box_transform=Box2BoxTransform(weights=(w1, w1, w2, w2)),
        cls_agnostic_bbox_reg=True,
        num_classes=NUM_CLASSES,
    )
    for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
]

mask_pooler = ROIPooler(
    output_size=14,
    scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
    sampling_ratio=0,
    pooler_type="ROIAlignV2",
)

mask_head = MaskRCNNConvUpsampleHead(
    input_shape=ShapeSpec(channels=256, width=14, height=14),
    num_classes=NUM_CLASSES,
    conv_dims=[256, 256, 256, 256, 256],
    conv_norm="LN",
)

cascade_roi_heads = CascadeROIHeads(
    num_classes=80,
    batch_size_per_image=512,
    positive_fraction=0.25,
    proposal_matchers=roi_head_matchers,
    box_in_features=["p2", "p3", "p4", "p5"],
    box_pooler=roi_pooler,
    box_heads=roi_box_heads,
    box_predictors=roi_box_predictors,
    mask_in_features=["p2", "p3", "p4", "p5"],
    mask_pooler=mask_pooler,
    mask_head=mask_head,
)

CascadeVITDetHuge = GeneralizedRCNN(
    backbone=backbone,
    proposal_generator=proposal_generator,
    roi_heads=cascade_roi_heads,
    pixel_mean=constants["imagenet_rgb256_mean"],  # torch.Tensor(constants['imagenet_rgb256_mean']),
    pixel_std=constants["imagenet_rgb256_std"],  # torch.Tensor(constants['imagenet_rgb256_std']),
    input_format="RGB",
)

# # update the model so now it contains the cascade roi heads, and the rest of the model is the same as the original vitdet model

# model.roi_heads = cascade_roi_heads
# model.backbone.net.embed_dim = 1280
# model.backbone.net.depth = 32
# model.backbone.net.num_heads = 16
# model.backbone.net.drop_path_rate = 0.5
# # 7, 15, 23, 31 for global attention
# model.backbone.net.window_block_indexes = list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))


# # clone the model so we can save it as cascade_vitdet_huge
# CascadeVITDetHuge = deepcopy(model)


if __name__ == "__main__":
    print(f"number of blocks in backbone: {len(CascadeVITDetHuge.backbone.net.blocks)}")
