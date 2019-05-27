# Detection-Fine-tuning-API
Because detection-models (Yolo, Mask-RCNN, etc) are often developed by the different frameworks and evaluated the performance on different hardware, it's hard to evaluate the performance (i.e, fps) and fine-tune these on your customized datasets. This project builds all detection-models with pytorch and provides the general API for training, fine-tuning and detection. 
## Installation
## Intro to Detection-Models
All detection models consist three components: low-level feature-extractor, high-level-extractor and detection. 
1. Low-level feature-extracor: it targets to extract the low-level features from images as same as classification and is often pretrained on ImageNet. In detection-models, we often use backbone-network to represent it.
2. High-level feature-extractor: it focuses on extracting features that are relevant to detection (i.e, 
Fine-grained features of roi). The common structure is Feature Pyramid Network (FPN) and this architecture is more capable to capture the object's information, both low-level and high-level. 
<p align="center">
  <img width="400" height="200" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/FPN.png>
</p>

3. Detection: it target to generate bounding boxes and their class-scores. <br/>
There are two main approaches in detection:
    1. Two-stage Detection: extracting all bbs that have objects first and then classifying each bb.
    2. One-stage Detection: extracting bbs and classifying bbs simultaneously.
## Fine-tuning
* Supervised Training (Given labeled data):
    1. [Fine-tuning the whole model.](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)
    2. Fine-tuning the high-level feature-extractor and detection.
* Unsupervised Training (Given unlabeled data or raw videos):
    <br/>Standard Fine-tuning Scheme: [Fine-tuning with detections.](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf)
    1. Fine-tuning the whole model with pseudo-bounding-boxes.
    2. Fine-tuning the high-level feature-extractor and detection with pseudo-bounding-boxes.  
## Supporting Models
* Two-stage Detection:
    1. [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870) inspired by [wkentaro](https://github.com/wkentaro/mask-rcnn.pytorch)
* One-stage Detection:
    1. [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002) inspired by [yhenon](https://github.com/yhenon/pytorch-retinanet)
    2. [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767) inspired by [ultralytics](https://github.com/ultralytics/yolov3) and [TommyAnqi](https://github.com/TommyAnqi/YOLOv3-Pytorch)
    3. [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533) inspired by [qijiezhao](https://github.com/qijiezhao/M2Det)
## Usage
## Details of Detection-models
### [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870)
#### Structure:
<p align="center">
  <img width="900" height="300" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/mask_rcnn_structure.png>
</p>

In Mask-RCNN, 
1. They use CNN layers (low-level feature-extractor and high-level feature-extractor) to extract the features from images first. 
2. They use RPN network to extract all bounding-box candidates. Each candidate has the corresponding coordinates (xmin, ymin, xmax, ymax) and the class (background or foreground). 
<p align="center">
  <img width="300" height="300" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/RPN.png>
</p>
3. They use FC to classify each candidate which is foregound.

### [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002)
#### Structure:
<p align="center">
  <img width="900" height="200" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/retinanet_structure.png>
</p>

### [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767)
#### Process Flow:
<p align="center">
  <img width="900" height="200" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/yolov3_flow.png>
</p>

#### Structure: 
<p align="center">
  <img width="900" height="600" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/yolov3_structure.png>
</p>

To analysis model-structure and data flow in detail, I plot the whole structure in the .pdf format ([pytorch_yolo](https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/pytorch_yolo.pdf)). In [pytorch_yolo](https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/pytorch_yolo.pdf), I mark scale-1 in blue, scale-2 in red and scale-3 in green.

To get different performances of yolo, we use two feature extractors in Yolov3:
1. Default feature extractor: [Darknet-53 or Darknet-19](https://github.com/ultralytics/yolov3)
2. Other feature extractor (more faster): [mobilenetv2](https://github.com/TommyAnqi/YOLOv3-Pytorch)

More details can be found in [A Closer Look at YOLOv3.](https://www.cyberailab.com/home/a-closer-look-at-yolov3)
### [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533)
<p align="center">
  <img width="900" height="200" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/m2det_structure.png>
</p>