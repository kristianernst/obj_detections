## Intention

The intention with this project is to build a robust object detection model.

First start is to simply build a model that makes bounding box predictions on images.
I decided to reference [detectron 2](https://github.com/facebookresearch/detectron2) heavily, 90%+ of the code is directly from there.

here is a direct link to their project folder on [VITDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)

When we get a satisfying result, ill start integrating it for video processing and tracking. That is for later tho.

Here are some good resources to learn from when referencing this hacked version of detectron2:

These papers are a good place to start:
www.arxiv.org/abs/2207.02696
www.arxiv.org/abs/2203.16527

we will use the following ROI heads:

mask rcnn: https://arxiv.org/abs/1703.06870

cascade rcnn: https://arxiv.org/abs/1712.00726, https://arxiv.org/abs/1906.09756

