from math import gcd
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from diffusers.utils import load_image
from PIL import Image

# # Load pipeline
# controlnet = FluxControlNetModel.from_pretrained(
#   "jasperai/Flux.1-dev-Controlnet-Upscaler",
#   torch_dtype=torch.bfloat16
# )
# pipe = FluxControlNetPipeline.from_pretrained(
#   "black-forest-labs/FLUX.1-dev",
#   controlnet=controlnet,
#   torch_dtype=torch.bfloat16
# )
# pipe.to("cuda")

# # Load a control image
# control_image = load_image(
#   "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/examples/input.jpg"
# )

# w, h = control_image.size

# # Upscale x4
# control_image = control_image.resize((w * 4, h * 4))

# image = pipe(
#     prompt="",
#     control_image=control_image,
#     controlnet_conditioning_scale=0.6,
#     num_inference_steps=28,
#     guidance_scale=3.5,
#     height=control_image.size[1],
#     width=control_image.size[0]
# ).images[0]
# image


PRETAINED_UPSCALER_PATH = "jasperai/Flux.1-dev-Controlnet-Upscaler"
PRETAINED_UPSCALER_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
DTYPES = torch.bfloat16


def get_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Upscaler:
    def __init__(
        self,
        controlnet_path: str = PRETAINED_UPSCALER_PATH,
        upscaler_model_path: str = PRETAINED_UPSCALER_MODEL_PATH,
        dtype: torch.dtype = DTYPES,
        device: str = "infer",
    ):
        self.controlnet = FluxControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)

        self.pipe = FluxControlNetPipeline.from_pretrained(upscaler_model_path, controlnet=self.controlnet, torch_dtype=dtype)

        self.device = get_device() if device == "infer" else device

        self.pipe.to(self.device)

    def __call__(self, image: Union[np.ndarray, torch.Tensor], max_dims: Tuple[int, int]):
        return self.forward(image, max_dims)

    def forward(self, image: Image.Image, max_dims: Tuple[int, int]):
        """
        Upscales an image to the maximum dimensions specified by max_dims.

        Args:
            image (Union[np.ndarray, torch.Tensor]): The image to upscale.
            max_dims (Tuple[int, int]): The maximum dimensions to upscale the image to, first dimension is width, second is height.

        Returns:
            Image: The upscaled image.
        """
        # if isinstance(image, np.ndarray):
        #     image = Image.fromarray(image)
        # elif isinstance(image, torch.Tensor):
        #     image = Image.fromarray(image.numpy())
        # else:
        #     raise ValueError(f"Invalid image type: {type(image)}")

        # now we need to ascertain how much we can upscale by with the AI model.
        # we do this by finding the max factor along either dimension.
        w, h = image.size
        factor = find_min_x(w, h, 8)
        print(f"factor: {factor}")
        image = image.resize((int(w * factor), int(h * factor)))
        print(f"upscaled image size: {image.size}")

        # now we can upscale the image
        image = self.pipe(
            prompt="",
            control_image=image,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=image.size[1],
            width=image.size[0],
        ).images[0]

        # now use cv2 to resize the image to the maximum dimensions specified by max_dims.
        # most precise resize
        # image = np.array(image)
        # # we need to get the aspect ratio and then find the max factor along either dimension.
        # aspect_ratio = w / h
        # max_factor = max(max_dims[0] / w, max_dims[1] / h)
        # dsize = (int(w * max_factor), int(h * max_factor))
        # image = cv2.resize(image, dsize, interpolation=cv2.INTER_LANCZOS4)

        return image


def lcm(a, b):
    """This function will find the least common multiple of a and b"""
    return a * b // gcd(a, b)


def find_min_x(a, b, z):
    """THis function will find the minimal multiplier for a and b that is divisible by z, and yields a zero remainder"""
    # Compute the minimal multipliers for a and b
    d1 = z // gcd(a, z)
    d2 = z // gcd(b, z)
    # Find the least common multiple of d1 and d2
    x = lcm(d1, d2)
    return x


if __name__ == "__main__":
    import os

    ASSETS_PATH = "assets/"
    TEST_IMAGES_PATH = os.path.join(ASSETS_PATH, "test_images")
    IN_PATH = os.path.join(TEST_IMAGES_PATH, "in")
    IN_IMG = os.path.join(IN_PATH, "venice.jpg")
    OUTPUT_IMAGES_PATH = os.path.join(ASSETS_PATH, "out")
    UPSCALED_TEST_IMAGE_PATH = os.path.join(OUTPUT_IMAGES_PATH, "upscaled")
    os.makedirs(UPSCALED_TEST_IMAGE_PATH, exist_ok=True)
    OUTPUT_IMAGE_PATH = os.path.join(UPSCALED_TEST_IMAGE_PATH, "upscaled.png")

    upscaler = Upscaler()

    # in_image = Image.open(IN_IMG)

    in_image = load_image(IN_IMG)
    print(f"type of in_image:{type(in_image)}")
    image = upscaler(in_image, max_dims=(1300, 800))
    image.save(OUTPUT_IMAGE_PATH)
