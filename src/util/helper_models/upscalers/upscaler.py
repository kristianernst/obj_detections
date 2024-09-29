from abc import ABCMeta, abstractmethod
from typing import Union

import cv2
import numpy as np
import PIL.Image as Image
from cv2.typing import MatLike
from pydantic import BaseModel
from torch import Tensor

# class Image(BaseModel):
#     image: Union[np.ndarray, Image.Image, Tensor]

#     class Config:
#         arbitrary_types_allowed = True


class Upscaler(metaclass=ABCMeta):
  """Generic upscaler class to be inherited from"""

  def __init__(self):
    """
    the `__init__` method of any subclass can specify its own set of arguments.
    """
    super().__init__()

  @abstractmethod
  def upscale(self, image):
    """All upscalers must have an upscale method

    returns:
        Image: upscaled image
    """
    pass


class CV2Upscaler(Upscaler):
  """
  Upscaler that uses cv2.dnn_superres.DnnSuperResImpl in the init method
  """

  def __init__(self):
    """
    the `__init__` method of any subclass can specify its own set of arguments.
    """
    super().__init__()

  def _init_model(self, model_path: str, name: str, scale: int):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(name, scale)
    return sr

  @staticmethod
  def _use_4x(img_shape: MatLike):
    """
    Determine if the image should be upscaled by 4x or 2x.

    Logic: if both dimensions are less than 1000 and one of them is less than 500, use 4x, otherwise use 2x
    """
    # get dimensions of MatLike object
    # h, w, c
    h, w, _ = img_shape

    if h < 500 or w < 500 and (h < 1000 and w < 1000):
      return True
    else:
      return False

  @abstractmethod
  def upscale(self, image):
    pass
