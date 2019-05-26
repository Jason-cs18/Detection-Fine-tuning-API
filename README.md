# Detection-Fine-tuning-API
Because detection-models (Yolo, Mask-RCNN, etc) are often developed by the different frameworks and evaluate the performance on different hardware, it's hard to evaluate the performance (i.e, fps). This project builds all detection-models on keras-tensorflow and provides the general API for training, fine-tuning and detection. 

## Supporting Models
1. Two-stage Models (Region proposals + classification):
..* [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870)
2. One-stage Models:
..* [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767)
..* [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533)
