import torch

from models.head.generators.anchor_generator import DefaultAnchorGenerator
from models.head.generators.box_regression import Box2BoxTransform
from models.head.generators.matcher import Matcher
from models.head.generators.proposal_generator import RPN, StandardRPNHead
from models.head.rcnn import GeneralizedRCNN
from models.head.roi.box_head import FastRCNNConvFCHead
from models.head.roi.fast_rcnn import FastRCNNOutputLayers
from models.head.roi.mask_head import MaskRCNNConvUpsampleHead
from models.head.roi.roi_heads import ROIHeads, StandardROIHeads
from models.layers import ShapeSpec
from models.poolers import ROIPooler
from util.vars import constants

# TODO lookup SFPN settings
NUM_CLASSES = 80


backbone = 1

standard_rpn_head = StandardRPNHead(
    in_channels=256, num_anchors=3, conv_dims=[-1, -1]
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


roi_head_matcher = Matcher(thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False)

roi_pooler = ROIPooler(
    output_size=7,
    scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
    sampling_ratio=0,
    pooler_type="ROIAlignV2",
)

roi_box_head = FastRCNNConvFCHead(
    input_shape=ShapeSpec(channels=256, height=7, width=7),
    conv_dims=[],
    fc_dims=[1024, 1024],
)

box2box_predictor_transform = Box2BoxTransform(weights=(10, 10, 5, 5))

box_predictor = FastRCNNOutputLayers(
    input_shape=ShapeSpec(channels=1024),
    test_score_thresh=0.05,
    box2box_transform=box2box_predictor_transform,
    num_classes=NUM_CLASSES,
)

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
)

roi_heads = StandardROIHeads(
    num_classes=80,
    batch_size_per_image=512,
    positive_fraction=0.25,
    proposal_matcher=roi_head_matcher,
    box_in_features=["p2", "p3", "p4", "p5"],
    box_pooler=roi_pooler,
    box_head=roi_box_head,
    box_predictor=box_predictor,
    mask_in_features=["p2", "p3", "p4", "p5"],
    mask_pooler=mask_pooler,
    mask_head=mask_head,
)


model = GeneralizedRCNN(
    backbone=backbone,
    proposal_generator=proposal_generator,
    roi_heads=roi_heads,
    pixel_mean=torch.Tensor(constants["imagenet_bgr256_mean"]),
    pixel_std=torch.Tensor(constants["imagenet_bgr256_std"]),
    input_format="BGR",
)


"""
from vitdet: https://github.com/facebookresearch/detectron2/blob/main/configs/common/models/mask_rcnn_vitdet.py
from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .mask_rcnn_fpn import model
from ..data.constants import constants

model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"

# Base
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "LN"

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

"""


"""
from https://github.com/facebookresearch/detectron2/blob/main/configs/common/models/mask_rcnn_fpn.py

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

from ..data.constants import constants

model = L(GeneralizedRCNN)(
    backbone=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="FrozenBN",
            ),
            out_features=["res2", "res3", "res4", "res5"],
        ),
        in_features="${.bottom_up.out_features}",
        out_channels=256,
        top_block=L(LastLevelMaxPool)(),
    ),
    proposal_generator=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    pixel_mean=constants.imagenet_bgr256_mean,
    pixel_std=constants.imagenet_bgr256_std,
    input_format="BGR",
)

"""
