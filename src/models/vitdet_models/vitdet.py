from functools import partial

import torch
import torch.nn as nn

# from models.head.fpn.mask_rcnn_fpn import model
from models.backbone.vit import SimpleFeaturePyramid, ViT
from models.head.generators.anchor_generator import DefaultAnchorGenerator
from models.head.generators.box_regression import Box2BoxTransform
from models.head.generators.matcher import Matcher
from models.head.generators.proposal_generator import RPN, StandardRPNHead
from models.head.rcnn import GeneralizedRCNN
from models.head.roi.box_head import FastRCNNConvFCHead
from models.head.roi.fast_rcnn import FastRCNNOutputLayers
from models.head.roi.mask_head import MaskRCNNConvUpsampleHead
from models.head.roi.roi_heads import StandardROIHeads
from models.layers import LastLevelMaxPool, ShapeSpec
from models.poolers import ROIPooler
from util.vars import constants

# model.pixel_mean = torch.Tensor(constants['imagenet_rgb256_mean'])
# model.pixel_std = torch.Tensor(constants['imagenet_rgb256_std'])
# model.input_format = "RGB"


embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1

IMG_SIZE = 1024
NET_OUT_FEATURES = "last_feat"

# TODO lookup SFPN settings
NUM_CLASSES = 80
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
  window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
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


roi_head_matcher = Matcher(thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False)

roi_pooler = ROIPooler(
  output_size=7,
  scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
  sampling_ratio=0,
  pooler_type="ROIAlignV2",
)

roi_box_head = FastRCNNConvFCHead(
  input_shape=ShapeSpec(channels=256, height=7, width=7),
  conv_dims=[256, 256, 256, 256],  # as per vitdet
  fc_dims=[IMG_SIZE],  # as per vitdet
  conv_norm="LN",  # as per vitdet
)

box2box_predictor_transform = Box2BoxTransform(weights=(10, 10, 5, 5))

box_predictor = FastRCNNOutputLayers(
  input_shape=ShapeSpec(channels=IMG_SIZE),
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
  conv_norm="LN",
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
  pixel_mean=constants["imagenet_rgb256_mean"],  # torch.Tensor(constants['imagenet_rgb256_mean']),
  pixel_std=constants["imagenet_rgb256_std"],  # torch.Tensor(constants['imagenet_rgb256_std']),
  input_format="RGB",
)


# TODO: load weights for the model!


if __name__ == "__main__":
  model.eval()
  # Define the batch size and input dimensions
  batch_size = 1
  height, width = IMG_SIZE, IMG_SIZE  # The expected image size

  # Create a dummy input tensor of shape (batch_size, 3, height, width)
  dummy_input = torch.randn(batch_size, 3, height, width)

  # Normalize the input using the same mean and std that the model expects
  # pixel_mean = torch.Tensor(constants['imagenet_rgb256_mean']).view(1, 3, 1, 1)
  # pixel_std = torch.Tensor(constants['imagenet_rgb256_std']).view(1, 3, 1, 1)

  # Normalize the input
  # normalized_input = (dummy_input - pixel_mean) / pixel_std

  # The model expects a list of dictionaries where each dictionary has an "image" key
  batched_inputs = [{"image": dummy} for dummy in dummy_input]

  # Run the forward pass
  with torch.no_grad():  # Disable gradient calculations for testing
    outputs = model(batched_inputs)

  # Print outputs for inspection
  print(outputs)
