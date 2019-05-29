# Detection-Fine-tuning-API
Because detection-models (Yolo, Mask-RCNN, etc) are often developed by the different frameworks and evaluated the performance on different hardware, it's hard to evaluate the performance (i.e, fps) and fine-tune these on your customized datasets. This project builds all detection-models with pytorch and provides the general API for training, fine-tuning and detection. 
#### Contents:
1. [Installation](#Installation)
2. [Intro to Detection-Models](#Intro-to-Detection-Models)
3. [Supporting Models](#Supporting-Models)
4. [Details of Detection-models](#Details-about-Detection-models)
5. [Loss visualization and analysis](#Loss-visualization-and-analysis)
6. [Fine-tuning](#Fine-tuning)
7. [Usage](#Usage)
    1. Mask-RCNN (in progress)
    2. RetinaNet (in progress)
    3. [Yolov3](#Yolov3)
    4. M2Det (in progress)
8. [References](#References)

## Installation
##### Clone and install requirements
    $ git clone https://github.com/jacksonly/Detection-Fine-tuning-API.git
    $ cd Detection-Fine-tuning-API/
    $ sudo pip3 install -r requirements.txt
    
##### Download pretrained weights (Yolov3 pretrained from coco)
    $ cd weights/
    $ bash ./yolov3/weights/download_weights.sh
Pretrained pytorch models (.pt): 
1. [yolov3.pt, yolov3-tiny.pt, yolov3-spp.pt](https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI)
    
## Intro to Detection-Models
All detection models consist three components: low-level feature-extractor, high-level-extractor and detection. 
<p align="center">
  <img width="900" height="200" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/Detect_Flow.png>
</p>

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
## Supporting Models
* Two-stage Detection:
    1. [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870) inspired by [wkentaro](https://github.com/wkentaro/mask-rcnn.pytorch)
* One-stage Detection:
    1. [RetinaNet (ICCV'17)](https://arxiv.org/abs/1708.02002) inspired by [yhenon](https://github.com/yhenon/pytorch-retinanet)
    2. [Yolov3 (arXiv'18)](https://arxiv.org/abs/1804.02767) inspired by [ultralytics](https://github.com/ultralytics/yolov3) and [TencentYoutuResearch](https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo)
    3. [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533) inspired by [qijiezhao](https://github.com/qijiezhao/M2Det)
## Details about Detection-models
### [Mask RCNN (CVPR'17)](https://arxiv.org/abs/1703.06870)
#### Structure:
<p align="center">
  <img width="700" height="300" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/mask_rcnn_structure.png>
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

To analysis model-structure and data flow in detail, I plot the whole structure in the .pdf format ([pytorch_yolo](https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/yolov3/pytorch_yolo.pdf)). In [pytorch_yolo](https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/yolov3/pytorch_yolo.pdf), I mark scale-1 in blue, scale-2 in red and scale-3 in green.

To get different performances of yolo, we use two feature extractors in Yolov3:
1. Default feature extractor: [Darknet-53 or Darknet-19](https://github.com/ultralytics/yolov3)
2. Other feature extractor (more faster): [mobilenetv2](https://github.com/TommyAnqi/YOLOv3-Pytorch)

More details can be found in [A Closer Look at YOLOv3.](https://www.cyberailab.com/home/a-closer-look-at-yolov3)
### [M2Det (AAAI'19, the latest SSD)](https://arxiv.org/abs/1811.04533)
#### Structure
<p align="center">
  <img width="900" height="200" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/m2det_structure.png>
</p>

## Loss visualization and analysis
To analysis the relationship between loss and bounding boxes, we plot the bbs that have small loss with green rectange and bbs that have large loss with purple rectange in pedestrian detection (dataset from WildTrack). We find that the hard bounding boxes are often having occlusion and hard to detect by models. 
<p align="center">
  <img width="900" height="500" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/loss_visualization.png>
</p>
Intuitively, we can use hard bbs as training set to get efficient training. But I found the performance is too bad when the number of hard bbs is small. Because detection-models need pair (image, bbs) as training set, only using subset of the all bbs may be easy to overfitting. ...

## Fine-tuning
### Problem
1. Class imbalance:</br>
    In fine-tuning, we must decide to use how many detections for backprogation because detection-models usually generate too many bounding boxes (many nosie) and the RoIs are overlapping with the small number of detections. If we use all detections as training sets., too many simple negatives will influence the performance of fine-tuning. Therefore, we need to design some strategies to resolve the imbalance problem between negative and positive samples. There are three main methods: Online hard example mining (OHEM), Focal Loss and Using positives only.
<p align="center">
  <img width="500" height="300" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/imbalance.png>
</p>

#### OHEM (Mask RCNN)
To get more efficient backpropagation on bounding boxes, we adopt the online hard example mininig (OHEM)  to training . In other words, we will sort all bounding boxes (positives and negatives) by loss in mini-batch and select B% bounding boxes that have the highest loss. Backprogation is performed based on the selected bounding boxes. Details can be refered in [paper](https://arxiv.org/pdf/1604.03540.pdf) and this method is often used to training two-stage detection-models.
#### Focal Loss (RetinaNet and Mask RCNN)
...
#### Using positives only (Yolov3 and M2Det)
In Yolov3, it only chooses the most suitable positive bounding boxes for backpagation. When they get the ground-truth bb1, they will search the image and find the detection bb2 that is the most possible candidate for bb1. Finally, they compute the loss between bb1 and bb2. This means each ground-truth only has one detection as candidate. Thus, there are not existing too many negatives.

* Supervised Training (Given labeled data):
    1. [Fine-tuning the whole model.](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)
    2. Fine-tuning the high-level feature-extractor and detection.
* Unsupervised Training (Given unlabeled data or raw videos):
    <br/>Standard Fine-tuning Scheme: [Fine-tuning with detections or Easy-to-Hard.](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf)
    1. Fine-tuning the whole model with pseudo-bounding-boxes.
    2. Fine-tuning the high-level feature-extractor and detection with pseudo-bounding-boxes.  

## Usage
### Yolov3:
#### 1. Detection (Yolov3)
We support to detect image and video:
1. Image: 
2. Video: 
#### 2. Data Prepartion (Yolov3)
1. Put all images to `./yolov3/data/custom/images/` and all labeled files to `./yolov3/data/custom/labels/`. Each image's name must be same as the corresponding labeled files.
An example image and label pair would be:
```
./yolov3/data/custom/images/00000000.png # image
./yolov3/data/custom/labels/00000000.txt # label
```
* One file per image (if no objects in image, no label file is required).
* One row per object.
* Each row is class x_center y_center width height format.
* Box coordinates must be in normalized xywh format (from 0 - 1).
* Classes are zero-indexed (start from 0).

An example label file with 32 persons (all class 0):
<p align="center">
  <img width="500" height="300" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/labeled_data.png>
</p>
2. Update the train.txt and val.txt in `./yolov3/data/custom/`
```
sada
```
3. Update the custom.names file in `./yolov3/data/custom/`
<p align="center">
  <img width="500" height="250" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/labeled_name.png>
</p>
4. Update the custom.data file in `./yolov3/data/custom/`
<p align="center">
  <img width="500" height="50" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/custom_data.png>
</p>
5. Update the custom.cfg file in `./yolov3/cfg/`. Each YOLO layer has 255 outputs: 85 outputs per anchor [4 box coordinates + 1 object confidence + 80 class confidences], times 3 anchors. If you use fewer classes, you can reduce this to [4 + 1 + n] * 3 = 15 + 3*n outputs, where n is your class count. This modification should be made to the output filter preceding each of the 3 YOLO layers. Also modify classes=80 to classes=n in each YOLO layer, where n is your class count. 
<p align="center">
  <img width="800" height="200" src=https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/change_structure.png>
</p>
#### 3. Fine-tuning (Yolov3)
We support 2 types of datasets:
1. Images with bounding boxes (Supervised Training).
``` bash
cd ./yolov3
# Train new model from scratch
python train.py --data ./data/custom/custom.data --cfg ./cfg/custom.cfg
# Fine-tune model from coco (Detection)
python train.py --data ./data/custom/custom.data --cfg ./cfg/custom.cfg --resume --class_num=1
# Fine-tune model from coco (High+Detection)
python train.py --data ./data/custom/custom.data --cfg ./cfg/custom.cfg --resume --class_num=1 --transfer_id=1
# Fine-tune model from coco (Low+High+Detection)
python train.py --data ./data/custom/custom.data --cfg ./cfg/custom.cfg --resume --class_num=1 --transfer_id=2
```
2. Images without bounding boxes (In progress). 
#### 4. Performance (Yolov3)
In experiment, I train yolov3 on pedestrain detection (from [WildTrack](https://cvlab.epfl.ch/data/data-wildtrack/)). The preprocessed data can be download in [images]() and [labels](). You can extract these and put to ./yolov3/
`from utils import utils; utils.plot_results()`<br/>
I plot the performance and loss. I find that the loss on class is getting convergence fast because single-class classification is more simple than multi-classification (in coco).

| Fine-tuning        | Performance  |    Loss     |
| ----------------------- |:-------------:|:-------------:|
| Train new model from scratch | ![](https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/1_performance.png)  |  ![](https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/1_loss.png) |
| Fine-tune model from coco (Detection) | ![](https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/2_performance.png)    |   ![](https://github.com/jacksonly/Detection-Fine-tuning-API/blob/master/images/2_loss.png) |
| Fine-tune model from coco (High+Detection)| are neat      |    $1 |
| Fine-tune model from coco (Low+High+Detection)| are neat      |    $1 |
## References:
[1] [Li Liu et al. Deep Learning for Generic Object Detection: A Survey. Arxiv 2018.](https://arxiv.org/pdf/1809.02165v1.pdf)
<br/>[2] [Kaiming He et al. Mask R-CNN. ICCV 2017.](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)
<br/>[3] [Tsung-Yi Lin et al. Focal Loss for Dense Object Detection. CVPR 2017](https://arxiv.org/pdf/1708.02002.pdf)
<br/>[4] [Joseph Redmon et al. YOLOv3: An Incremental Improvement. Arxiv 2018.](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
<br/>[5] [Qijie Zhao et al. M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network. AAAI 2019.](https://arxiv.org/pdf/1811.04533.pdf)
<br/>[6] [Abhinav Shrivastava et al. Training Region-based Object Detectors with Online Hard Example Mining. CVPR 2016.](https://arxiv.org/pdf/1604.03540.pdf)
<br/>[7] [Yang Zou et al. Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training. ECCV 2018.](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf)
<br/>[8] [SouYoung Jin et al. Unsupervised Hard Example Mining from Videos for Improved Object Detection. ECCV 2018.](http://vis-www.cs.umass.edu/unsupVideo/docs/unsup-video_eccv2018.pdf)
<br/>[9] [Paul Voigtlaender et al. Large-Scale Object Discovery and Detector Adaptation from Unlabeled Video. CVPR 2018.](https://arxiv.org/pdf/1712.08832.pdf)