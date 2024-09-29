from models.head.generators.box_regression import Box2BoxTransform
from models.head.generators.matcher import Matcher
from models.head.roi.box_head import FastRCNNConvFCHead
from models.head.roi.fast_rcnn import FastRCNNOutputLayers
from models.head.roi.mask_head import MaskRCNNConvUpsampleHead
from models.head.roi.roi_heads import CascadeROIHeads
from models.layers import ShapeSpec
from models.poolers import ROIPooler
from models.vitdet_models.vitdet import NUM_CLASSES, model

# rm args that does not exist for cascade version of vitdet
# [model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

roi_pooler = ROIPooler(
  output_size=7,
  scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
  sampling_ratio=0,
  pooler_type="ROIAlignV2",
)

roi_head_matchers = [Matcher(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False) for th in [0.5, 0.6, 0.7]]

roi_box_heads = [
  FastRCNNConvFCHead(input_shape=ShapeSpec(channels=256, height=7, width=7), conv_dims=[256, 256, 256, 256], fc_dims=[1024], conv_norm="LN")
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

model.roi_heads = cascade_roi_heads
