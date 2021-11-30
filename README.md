# Attentive Prototypes for Source-free Unsupervised Domain Adaptive 3D Object Detection

PyTorch code release of the paper "Attentive Prototypes for Source-free Unsupervised Domain Adaptive 3D Object Detection",

by [Deepti Hegde](https://deeptibhegde.github.io), [Vishal Patel](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)

![image](/imgs/main_final_2.drawio.jpg)


(Currently has instructions for model inference and evaluation only, training steps to be updated soon.)

Follow the instructions for installation and implementation of the method for each base object detection network in the respective folders [SECOND-iou](/SECOND-iou/) and [PointRCNN](/PointRCNN/)

# Dataset preperation

1. Download the relevant datasets: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) , [Waymo](https://waymo.com/intl/en_us/dataset-download-terms/) , [nuScenes](https://www.nuscenes.org/download)

2. Organize each folder inside [data](/data/) like the following


```
AttentivePrototypeSFUDA

├── data (main data folder)
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
|
|
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
|
|
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── pcdet_gt_database_train_sampled_xx/
│   │   │── pcdet_waymo_dbinfos_train_sampled_xx.pkl  
|
|
├── PointRCNN
|   ├── data (link to main data folder)
|   ├── pointrcnn_attention
├── SECOND-iou
|   ├── data (link to main data folder)
|   ├── pcdet
|   ├── tools
```


We implement the proposed method for two object detectors, [SECOND-iou](/secondiou/) and [PointRCNN](/pointrcnn/) for several domain shift scenarios. You can find the folder of pretrained models [here](https://drive.google.com/drive/folders/1nbh1LdYdNdinYg0xP4MPreW-RkaGznNE?usp=sharing). Find specific model downloads and their corresponding config files below.


| SECOND-iou |
-------------------------------------------------
| Domain shif | Model file  | Configuration file |
| ----------- | ----------- | -------------------|
| Waymo  -> KITTI| [download](https://drive.google.com/file/d/1X58-Lfiyv9j8ORycQLXEWyYox4SYFuVt/view?usp=sharing) | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |
|  Waymo  -> nuScenes | [download](https://drive.google.com/file/d/1HWpPGZLuB76C979QnfjWDJOGG_0_PhaS/view?usp=sharing)        | [link](SECOND-iou/tools/cfgs/nuscenes_models/secondiou_car_oracle.yaml) |
|  nuScenes -> KITTI| [download](https://drive.google.com/file/d/1QSPyY8FjgjbMw1GlVmpGx6RThP2bQXDK/view?usp=sharing)        | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |




| PointRCNN |
--------------------------------------------------
| Domain shif | Model file  | Configuration file |
| ----------- | ----------- | -------------------|
| Waymo  -> KITTI| [download](https://drive.google.com/file/d/1O4bhgdlqkgYIXY2aEYRLejbWnWwPbvfv/view?usp=sharing)       | [link](PointRCNN/pointrcnn_attention/tools/cfgs/default.yaml) |
|  KITTI  -> nuScenes | [download](https://drive.google.com/file/d/1bvdmnSQPEds2St9g7QJnJkkcUyY2Ye8V/view?usp=sharing)        | [link](PointRCNN/pointrcnn_attention/tools/cfgs/default.yaml) |
|  nuScenes -> KITTI| [download](https://drive.google.com/file/d/1VptyJk3j62s22IP_3R9SzBoWhErz07Ov/view?usp=sharing)        | [link](PointRCNN/pointrcnn_attention/tools/cfgs/default.yaml) |


Follow the instructions to implement the method in the folders [SECOND-iou](SECOND-iou/README.md) and [PointRCNN](PointRCNN/README.md)

