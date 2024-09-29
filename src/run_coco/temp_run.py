import os

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torch import nn
from tqdm import tqdm

from checkpoint.detection_checkpoint import DetectionCheckpointer
from data import transforms as T
from data.catalog import DatasetCatalog, MetadataCatalog
from data.coco_data.build import (build_detection_test_loader,
                                  build_detection_train_loader,
                                  get_detection_dataset_dicts)
from data.coco_data.coco_util import register_coco_instances
from data.coco_data.evaluation import COCOEvaluator
from data.dataset_mapper import DatasetMapper
from data.evaluator import inference_on_dataset
# from models.vitdet_models.cascade_vitdet import model as CascadeVITDet
from models.vitdet_models.cascade_vitdet_huge import CascadeVITDetHuge
from models.vitdet_models.vitdet import model as VITDet
from util.images import resize_image
from util.load_weights import load_model
from util.visualizer import Visualizer

"""
we need to make a version of this that does not use LazyCall

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
"""


DATASET_PATH = "datasets"
COCO_PATH = os.path.join(DATASET_PATH, "coco")
ANNOTATIONS_PATH = os.path.join(COCO_PATH, "annotations")
TRAIN_2017_JSON = os.path.join(ANNOTATIONS_PATH, "instances_train2017.json")
TRAIN_2017_IMAGES = os.path.join(COCO_PATH, "train2017")
VAL_2017_JSON = os.path.join(ANNOTATIONS_PATH, "instances_val2017.json")
VAL_2017_IMAGES = os.path.join(COCO_PATH, "val2017")

register_coco_instances("coco_2017_train", {}, TRAIN_2017_JSON, TRAIN_2017_IMAGES)
register_coco_instances("coco_2017_val", {}, VAL_2017_JSON, VAL_2017_IMAGES)
test_metadata = MetadataCatalog.get("coco_2017_val")

# Debug print to check registered datasets
print("Registered datasets:", DatasetCatalog.list())


train_mapper = DatasetMapper(
  is_train=True,
  augmentations=[
    T.ResizeShortestEdge(
      short_edge_length=(640, 672, 704, 736, 768, 800),
      sample_style="choice",
      max_size=1333,
    ),
    T.RandomFlip(horizontal=True),
  ],
  image_format="BGR",
  use_instance_mask=True,
)

test_mapper = DatasetMapper(
  is_train=False,
  augmentations=[
    T.ResizeShortestEdge(short_edge_length=800, max_size=1333),  # are not currently used
    #  T.ResizeShortestEdge(short_edge_length=5000, max_size=5000)
  ],
  image_format="BGR",
)


# dataloader = OmegaConf.create()
# dataloader_train = build_detection_train_loader(
#     dataset=get_detection_dataset_dicts(names="coco_2017_train"),
#     mapper=train_mapper,
#     total_batch_size=16,
#     num_workers=4,
# )

dataloader_test = build_detection_test_loader(
  dataset=get_detection_dataset_dicts(names="coco_2017_val", filter_empty=False),
  mapper=test_mapper,
  num_workers=1,
)

dataloader_evaluator = COCOEvaluator(
  dataset_name="coco_2017_val",
)


# def evaluate_model(model: nn.Module, dataloader, evaluator):
#     results = inference_on_dataset(model=model, data_loader=dataloader, evaluator=evaluator)
#     print(results)
#     return results


# # Visualize function
# def visualize_sample(model, data_loader, num_samples=1):

#     # Set the model to evaluation mode

#     model.eval()
#     # Iterate over the data loader and visualize predictions
#     for idx, inputs in enumerate(data_loader):
#         if idx >= num_samples:
#             break

#         # Run inference
#         with torch.no_grad():
#             outputs = model(inputs)[0]

#         # Create a visualizer
#         #print(f"inputs: {inputs}")
#         print(f"inputs['image'].shape: {inputs[0]['image'].shape}")
#         # so the tensor has the following shape:
#         #print(f"inputs shape: {inputs.sh}")
#         # H, W, C with RGB format.
#         img_conv = torch.einsum("chw->hwc", inputs[0]["image"]).to("cpu")
#         print(f"img_conv.shape: {img_conv.shape}")
#         v = Visualizer(img_conv, test_metadata, scale=1.2)

#         # print(f"outputs: {outputs}")
#         print(f"type of outputs: {type(outputs)}")
#         # print(f"len of outputs: {len(outputs)}")

#         for box in outputs["instances"].pred_boxes.to("cpu"):
#             v.draw_box(box)
#             v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))

#         v = v.get_output()
#         img = v.get_image()[:, :, ::-1] # from BGR to RGB
#         plt.imshow(img)
#         plt.show()

if __name__ == "__main__":
  # make a simple eval loop
  import pickle

  import cv2
  import numpy as np
  import torch
  import tqdm
  from PIL import Image

  # device = torch.device(
  #     "cuda" if torch.cuda.is_available()
  #     else "mps" if torch.backends.mps.is_available()
  #     else "cpu"
  # )

  device = torch.device("cpu")
  print(f"Using device: {device}")
  print("Loading model...")
  WEIGHTS_PATH = os.path.join("models", "weights")

  model_names = ["vitdet_base", "vitdet_cascade_base", "vitdet_cascade_huge"]
  model_name = model_names[2]

  base_weights = (
    os.path.join(WEIGHTS_PATH, "model_final_61ccd1.pkl")
    if model_name == "vitdet_base"
    else os.path.join(WEIGHTS_PATH, "vitdet_cascade_base.pkl")
    if model_name == "vitdet_cascade_base"
    else os.path.join(WEIGHTS_PATH, "vitdet_cascade_huge.pkl")
    if model_name == "vitdet_cascade_huge"
    else None
  )

  if model_name is None:
    raise ValueError(f"Model name {model_name} not found, check if the model is either of {', '.join(model_names)}")

  model = (
    VITDet
    if model_name == "vitdet_base"
    # else CascadeVITDet
    # if model_name == "vitdet_cascade_base"
    else CascadeVITDetHuge
    if model_name == "vitdet_cascade_huge"
    else None
  )

  if model is None:
    raise ValueError(f"Model {model_name} not found, check if the model is either of {', '.join(model_names)}")

  # checkpointer = DetectionCheckpointer(model)
  # checkpointer.load(base_weights)

  model = load_model(model, base_weights)
  model.eval()
  model.to(device)

  max_width = 1300
  max_height = 800
  use_one_img = True
  asset_dir = "assets"
  test_images_dir = os.path.join(asset_dir, "test_images")
  in_dir = os.path.join(test_images_dir, "in")
  out_dir = os.path.join(test_images_dir, "out", model_name)
  os.makedirs(out_dir, exist_ok=True)

  SENSITIVITY_THRESH = 0.6

  # filename = "b.jpg"
  filenames = os.listdir(in_dir)
  filenames = [filename for filename in filenames if filename.endswith(".jpg")]

  # tqdm

  for filename in tqdm.tqdm(filenames):
    image_path = os.path.join(in_dir, filename)
    image = Image.open(image_path)
    print(f"processing image: {filename}")
    # find the aspect ratio
    image = np.array(image, dtype=np.uint8)
    # ive found that the max width is around 1000, so we need to scale the width, and the height needs to be scaled to height * aspect_ratio
    image = resize_image(image, max_width, max_height)
    image = np.moveaxis(image, -1, 0)  # the model expects the image to be in channel first format
    if use_one_img:
      images = [image]
    else:
      # if image is very wide, split into 2.
      c, img_height, img_width = image.shape
      img_width_half = img_width // 2
      image_1 = image[:, :, :img_width_half]
      image_2 = image[:, :, img_width_half:]
      images = [image_1, image_2]

    with torch.inference_mode():
      for i, img in enumerate(images):
        output = model([{"image": torch.from_numpy(img).to(device)}])
        # img is a np array, we need to convert it to a tensor
        img_tensor = torch.from_numpy(img)
        img_conv = torch.einsum("chw->hwc", img_tensor)
        img_conv = img_conv.numpy()
        # img_conv = torch.einsum("chw->hwc", img_tensor)
        visualizer = Visualizer(img_conv, metadata=test_metadata, scale=1.2)
        instances = output[0]["instances"].to("cpu")
        # only take the instances with a score greater than 0.9 and only the first 4 instances
        instances = instances[instances.scores > SENSITIVITY_THRESH]  # [:10]

        visualized_output = visualizer.draw_instance_predictions(instances)
        if len(images) > 1:
          out_filename = f"{filename.split('.')[0]}_out_{i}.png"
        else:
          out_filename = f"{filename.split('.')[0]}_out.png"
        visualized_output.save(os.path.join(out_dir, out_filename))
      # # model = VITDet
  # # does not seem to work
  # # model.roi_heads.box_predictor.test_score_thresh = 0.7
  # # model.roi_heads.num_classes = 1
  # # model.roi_heads.mask_head.num_classes = 1
  # # model.roi_heads.box_predictor.num_classes = 1

  # # model = load_model(model, base_weights)

  # print("Evaluating model...")

  # #evaluate_model(model, dataloader_test, dataloader_evaluator)
  # visualize_sample(model, dataloader_test, num_samples=1)
