## intention


The intention with this project is to build a robust object detection model. 

First start is to simply build a model that makes bounding box predictions on images.

When we get a satisfying result, ill start integrating it for video processing and tracking. That is for later tho.


Ideally, id like to learn a lot from this project - 

* get a good understanding of current best practices.
These papers are a good place to start: 
www.arxiv.org/abs/2207.02696
www.arxiv.org/abs/2203.16527

VITDet: https://github.com/pytorch/vision/pull/7690


HF implementation: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/vitdet/modeling_vitdet.py


At the moment, only the backbone is available.



MAE: https://arxiv.org/pdf/2111.06377 (pretrained vision transformer)