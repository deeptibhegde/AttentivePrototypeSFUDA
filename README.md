# Attentive Prototypes for Source-free Unsupervised Domain Adaptive 3D Object Detection

PyTorch code release of the paper "Attentive Prototypes for Source-free Unsupervised Domain Adaptive 3D Object Detection",

![image](/imgs/main_final_2.drawio.jpg)


(Currently supports model inference and evaluation only, training steps to be updated soon.)


# Dataset preperation

1. Download the relevant datasets: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) , [Waymo](https://waymo.com/intl/en_us/dataset-download-terms/) , [nuScenes](https://www.nuscenes.org/download)

2. Organize each folder inside [data](/data/) like the following


```
AttentivePrototypeSFUDA

├── data (main data folder)```
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


We implement the proposed method for two object detectors, [SECOND-iou](/secondiou/) and [PointRCNN](/pointrcnn/) for several domain shift scenarios. You can find the folder of pretrained models [here](). Find specific model downloads and their corresponding config files below.


| SECOND-iou |
-------------------------------------------------
| Domain shif | Model file  | Configuration file |
| ----------- | ----------- | -------------------|
| Waymo  -> KITTI| [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dhegde1_jh_edu/EaF3HPR91A5OqRnTCD3sKw4Bw0HbWHVaF3mRrBdM9ybS-g?e=f9UurE)       | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |
|  Waymo  -> nuScenes | [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dhegde1_jh_edu/EaF3HPR91A5OqRnTCD3sKw4Bw0HbWHVaF3mRrBdM9ybS-g?e=f9UurE)        | [link](SECOND-iou/tools/cfgs/nuscenes_models/secondiou_car_oracle.yaml) |
|  nuScenes -> KITTI| [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dhegde1_jh_edu/EUC7cRbRwuNOp11bUqXhOXgB5uGwuBmF1zP0R8cW2BPZJQ?e=qduaqy)        | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |




| PointRCNN |
--------------------------------------------------
| Domain shif | Model file  | Configuration file |
| ----------- | ----------- | -------------------|
| Waymo  -> KITTI| [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dhegde1_jh_edu/EWS-zL0tDItOurHL8DV18AgB92kQDBRcocIJ3PpuDwSamw?e=Zl4dtm)       | [download](PointRCNN/pointrcnn_attention/tools/cfgs/default.yaml) |
|  KITTI  -> nuScenes | [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dhegde1_jh_edu/ERAV9hreSSBGqLIFXD7xsB0B8zNaE6CDjlGHYnuKXZbBWw?e=25r0d6)        | [link](PointRCNN/pointrcnn_attention/tools/cfgs/default.yaml) |
|  nuScenes -> KITTI| [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/dhegde1_jh_edu/EQJ-IusineZLvjpjo5iOJ7ABwPUZ-Mh_mYVrODx8lPX6Eg?e=eAnf0c)        | [link](PointRCNN/pointrcnn_attention/tools/cfgs/default.yaml) |


Follow the instructions to implement the method in [SECOND-iou](SECOND-iou/README.md) and [PointRCNN](PointRCNN/README.md)


