# Detection-Fine-tuning-API
Because detection-models (Yolo, Mask-RCNN, etc) are often developed by the different frameworks and evaluated the performance on different hardware, it's hard to evaluate the performance (i.e, fps) and fine-tune these on your customized datasets. This project builds all detection-models with pytorch and provides the general API for training, fine-tuning and detection. 
## Installation
## Supporting Fine-tuning
* Supervised Training (Given labeled data):
    1. [Fine-tuning the whole model.](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)
    2. Fine-tuning the detection.
* Unsupervised Training (Given unlabeled data or raw videos):
    <br/>Standard Fine-tuning Scheme: [Fine-tuning with detections.](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf)
    1. Fine-tuning the whole model with pseudo-bounding-boxes.
    2. Fine-tuning detection with pseudo-bounding-boxes. 
## Supporting Models
All detection models consist two components: feature extractor and detection. In detection, there are two main approaches: 
* Two-stage Detection (extracting all bbs that have object and then classifying each bb):
    1. [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870) inspired by [wkentaro](https://github.com/wkentaro/mask-rcnn.pytorch)
* One-stage Detection (extracting bbs and classifying bbs simultaneously):
    1. [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002) inspired by [yhenon](https://github.com/yhenon/pytorch-retinanet)
    2. [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767) inspired by [ultralytics](https://github.com/ultralytics/yolov3) and [TommyAnqi](https://github.com/TommyAnqi/YOLOv3-Pytorch)
    3. [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533) inspired by [M2Det](https://github.com/qijiezhao/M2Det)
## Usage
## Analysis of Detection-models
### [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870)
### [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002)
### [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767)
Process Flow:<br/>
<p align="center">
  <img width="900" height="200" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/yolov3_flow.png>
</p>
Structure: <br/>
<p align="center">
  <img width="900" height="200" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/yolov3_structure.png>
</p>
We use two feature extractors in Yolov3:
1. Default feature extractor: [Darknet-53 or Darknet-19](https://github.com/ultralytics/yolov3)
2. Other feature extractor (more faster): [mobilenetv2](https://github.com/TommyAnqi/YOLOv3-Pytorch)
### [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533)