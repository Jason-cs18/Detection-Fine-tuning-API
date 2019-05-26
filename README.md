# Detection-Fine-tuning-API
Because detection-models (Yolo, Mask-RCNN, etc) are often developed by the different frameworks and evaluated the performance on different hardware, it's hard to evaluate the performance (i.e, fps) and fine-tune these on your customized datasets. This project builds all detection-models on keras-tensorflow and provides the general API for training, fine-tuning and detection. 
## Installation
## Supporting Models
* Two-stage Models (Region proposals + classification):
    1. [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870) inspired by [Mask_RCNN](https://github.com/matterport/Mask_RCNN)
* One-stage Models:
    1. [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002) inspired by [keras-retinanet](https://github.com/fizyr/keras-retinanet)
    2. [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767) inspired by [keras-yolo3](https://github.com/qqwweee/keras-yolo3) and [mobilenetv2-yolov3](https://github.com/fsx950223/mobilenetv2-yolov3)
    3. [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533) inspired by [M2Det](https://github.com/qijiezhao/M2Det) and [m2det-tensorflow](https://github.com/tadax/m2det)
## Usage
## Analysis of Detection-models
### [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870)
### [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002)
### [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767)
Structure:<br/>
![alt text](https://github.com/jacksonly/Detection-Fine-tuning-API/tree/master/Images/yolov3.png)

We use two feature extractors in Yolov3:
1. Default feature extractor: [Darknet-53](https://github.com/qqwweee/keras-yolo3)
2. Other feature extractor (more faster): [mobilenetv2](https://github.com/fsx950223/mobilenetv2-yolov3)
### [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533)