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




