# Detection-Fine-tuning-API
Because detection-models (Yolo, Mask-RCNN, etc) are often developed by the different frameworks and evaluated the performance on different hardware, it's hard to evaluate the performance (i.e, fps) and fine-tune these on your customized datasets. This project builds all detection-models on keras-tensorflow and provides the general API for training, fine-tuning and detection. 

## Supporting Models
* Two-stage Models (Region proposals + classification):
    1. [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870)
* One-stage Models:
    1. [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002)
    2. [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767)
    3. [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533)
