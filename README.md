# Detection-Fine-tuning-API
Because detection-models (Yolo, Mask-RCNN, etc) are often developed by the different frameworks and evaluated the performance on different hardware, it's hard to evaluate the performance (i.e, fps) and fine-tune these on your customized datasets. This project builds all detection-models on keras-tensorflow and provides the general API for training, fine-tuning and detection. 
## Supporting Models
* Two-stage Models (Region proposals + classification):
    1. [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870)
* One-stage Models:
    1. [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002)
    2. [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767)
    3. [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533)
### [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870)
### [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002)
### [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767)
Structure: 
![alt text](https://github.com/jacksonly/Detection-Fine-tuning-API/Images/yolov3.png)
We test different feature extractors in Yolov3:
    1. Default feature extractor: [Darknet-53](https://github.com/qqwweee/keras-yolo3)
    2. Other feature extractor: [mobilenetv2](https://github.com/fsx950223/mobilenetv2-yolov3)
### [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533)