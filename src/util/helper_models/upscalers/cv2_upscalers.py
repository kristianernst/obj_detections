import logging
import os

import cv2
from cv2.typing import MatLike

from util.helper_models.upscalers.upscaler import CV2Upscaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EDSR(CV2Upscaler):
    def __init__(self, model_path_4x, model_path_2x):
        self.model_path_4x = model_path_4x
        self.model_path_2x = model_path_2x
        self.model_4x = self._init_model(self.model_path_4x, name="edsr", scale=4)
        self.model_2x = self._init_model(self.model_path_2x, name="edsr", scale=2)
        super().__init__()

    def upscale(self, image: MatLike):
        if self._use_4x(image.shape):
            logger.info("Upscaling by 4x")
            return self.model_4x.upsample(image)
        else:
            logger.info("Upscaling by 2x")
            return self.model_2x.upsample(image)


class ESPCN(CV2Upscaler):
    def __init__(self, model_path_4x, model_path_2x):
        self.model_path_4x = model_path_4x
        self.model_path_2x = model_path_2x
        self.model_4x = self._init_model(self.model_path_4x, name="espcn", scale=4)
        self.model_2x = self._init_model(self.model_path_2x, name="espcn", scale=2)
        super().__init__()

    def upscale(self, image: MatLike):
        if self._use_4x(image.shape):
            logger.info("Upscaling by 4x")
            return self.model_4x.upsample(image)
        else:
            logger.info("Upscaling by 2x")
            return self.model_2x.upsample(image)


if __name__ == "__main__":
    # model
    UTIL_DIR = "util"
    HELPER_MODELS_DIR = os.path.join(UTIL_DIR, "helper_models")
    UPSCALERS_DIR = os.path.join(HELPER_MODELS_DIR, "upscalers")
    MODEL_FILES_DIR = os.path.join(UPSCALERS_DIR, "modelfiles")
    UPSCALER_EDSR_2x = "EDSR_x2.pb"
    UPSCALER_EDSR_PATH2x = os.path.join(MODEL_FILES_DIR, UPSCALER_EDSR_2x)
    UPSCALER_EDSR_4x = "EDSR_x4.pb"
    UPSCALER_EDSR_PATH4x = os.path.join(MODEL_FILES_DIR, UPSCALER_EDSR_4x)

    UPSCALER_ESPCN_2x = "ESPCN_x2.pb"
    UPSCALER_ESPCN_PATH2x = os.path.join(MODEL_FILES_DIR, UPSCALER_ESPCN_2x)
    UPSCALER_ESPCN_4x = "ESPCN_x4.pb"
    UPSCALER_ESPCN_PATH4x = os.path.join(MODEL_FILES_DIR, UPSCALER_ESPCN_4x)

    # in
    ASSETS_DIR = "assets"
    TEST_IMAGES_DIR = os.path.join(ASSETS_DIR, "test_images")
    IN_DIR = os.path.join(TEST_IMAGES_DIR, "in")
    IMG_PATH = os.path.join(IN_DIR, "venice.jpg")

    # out
    OUT_DIR = os.path.join(TEST_IMAGES_DIR, "out")
    UPSCALED_DIR = os.path.join(OUT_DIR, "upscaled")
    os.makedirs(UPSCALED_DIR, exist_ok=True)
    UPSCALED_IMG_PATH_EDSR = os.path.join(UPSCALED_DIR, "venice_upscaled_EDSR.jpg")
    UPSCALED_IMG_PATH_ESPCN = os.path.join(UPSCALED_DIR, "venice_upscaled_ESPCN.jpg")
    img = cv2.imread(IMG_PATH)

    # run
    import time

    start_time = time.time()
    upscaler_EDSR = EDSR(model_path_4x=UPSCALER_EDSR_PATH4x, model_path_2x=UPSCALER_EDSR_PATH2x)
    result_EDSR = upscaler_EDSR.upscale(img)
    cv2.imwrite(UPSCALED_IMG_PATH_EDSR, result_EDSR)
    end_time = time.time()
    print(f"Time taken for EDSR: {end_time - start_time} seconds")

    start_time = time.time()
    upscaler_ESPCN = ESPCN(model_path_4x=UPSCALER_ESPCN_PATH4x, model_path_2x=UPSCALER_ESPCN_PATH2x)
    result_ESPCN = upscaler_ESPCN.upscale(img)
    cv2.imwrite(UPSCALED_IMG_PATH_ESPCN, result_ESPCN)
    end_time = time.time()
    print(f"Time taken for ESPCN: {end_time - start_time} seconds")
