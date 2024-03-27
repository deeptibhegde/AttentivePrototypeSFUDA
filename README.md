# Attentive Prototypes for Source-free Unsupervised Domain Adaptive 3D Object Detection

PyTorch code release of the paper "Attentive Prototypes for Source-free Unsupervised Domain Adaptive 3D Object Detection",

by [Deepti Hegde](https://deeptibhegde.github.io), [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)

[[arXiv]](https://arxiv.org/abs/2111.15656)

![image](/imgs/main_final_2.drawio.jpg)



Follow the instructions for installation from [ST3D](https://github.com/CVMI-Lab/ST3D)

# Dataset preperation

1. Download the relevant datasets: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) , [Waymo](https://waymo.com/intl/en_us/dataset-download-terms/) , [nuScenes](https://www.nuscenes.org/download)

2. Generate the pickle files for each datset

  - KITTI 
      
      ```
      python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
      ```
  
  
  - nuScenes
  
      ```
      python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml  --version v1.0-trainval
      ```
  
  - Waymo (this will take several hours for the whole dataset. You may download only a subset for a faster pre-processing and source training)
  
      ```
      python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos  --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
      ```

3. Organize each folder inside [data](/data/) like the following


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
    
    ```
## The below instructions are deprecated, please skip to [this section]([#Training](https://github.com/deeptibhegde/AttentivePrototypeSFUDA/edit/main/README.md#training))  for up-to-date instructions.

We implement the proposed method for two object detectors, [SECOND-iou](/secondiou/) and [PointRCNN](/pointrcnn/) for several domain shift scenarios. You can find the folder of pretrained models [here](https://drive.google.com/drive/folders/1nbh1LdYdNdinYg0xP4MPreW-RkaGznNE?usp=sharing). Find specific model downloads and their corresponding config files below.


| SECOND-iou |
-------------------------------------------------
| Domain shift | Model file  | Configuration file |
| ----------- | ----------- | -------------------|
| Waymo  -> KITTI| [download](https://drive.google.com/file/d/1X58-Lfiyv9j8ORycQLXEWyYox4SYFuVt/view?usp=sharing) | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |
|  Waymo  -> nuScenes | [download](https://drive.google.com/file/d/1HWpPGZLuB76C979QnfjWDJOGG_0_PhaS/view?usp=sharing)        | [link](SECOND-iou/tools/cfgs/nuscenes_models/secondiou_car_oracle.yaml) |
|  nuScenes -> KITTI| [download](https://drive.google.com/file/d/1QSPyY8FjgjbMw1GlVmpGx6RThP2bQXDK/view?usp=sharing)        | [link](SECOND-iou/tools/cfgs/kitti_models/secondiou_oracle_ros.yaml) |




| PointRCNN |
--------------------------------------------------
| Domain shift | Model file  | Configuration file |
| ----------- | ----------- | -------------------|
| Waymo  -> KITTI| [download](https://drive.google.com/file/d/1O4bhgdlqkgYIXY2aEYRLejbWnWwPbvfv/view?usp=sharing)       | [link](PointRCNN/pointrcnn_attention/tools/cfgs/default.yaml) |
|  KITTI  -> nuScenes | [download](https://drive.google.com/file/d/1bvdmnSQPEds2St9g7QJnJkkcUyY2Ye8V/view?usp=sharing)        | [link](PointRCNN/pointrcnn_attention/tools/cfgs/default.yaml) |
|  nuScenes -> KITTI| [download](https://drive.google.com/file/d/1VptyJk3j62s22IP_3R9SzBoWhErz07Ov/view?usp=sharing)        | [link](PointRCNN/pointrcnn_attention/tools/cfgs/default.yaml) |


Follow the instructions to implement the method in the folders [SECOND-iou](SECOND-iou/README.md) and [PointRCNN](PointRCNN/README.md)




# Training

The entire training procedure may be divided into two stages: 1) source model training and 2) source-free domain adaptation. 

## Source model training

### Single GPU training
  ```
    python train.py --cfg_file cfgs/da-waymo-kitti_models/secondiou/secondiou_cyc.yaml --extra_tag {PREFERRED NAME}
  ```
### Multi-GPU training
  ```
    bash scripts/dist_train.sh {NUM_GPUS} --cfg_file cfgs/da-waymo-kitti_models/secondiou/secondiou_cyc.yaml --extra_tag {PREFERRED NAME}
  ```

## Source-free domain adaptation

Choose the best performing model from the previous step and use it as the source trained model. 
  ```
    python train.py --cfg_file cfgs/da-waymo-kitti_models/secondiou_attproto/secondiou_proto_ros_cyc.yaml --extra_tag {PREFERRED NAME} --pretrained_model {SOURCE_MODEL_PATH}
  ```

# Testing

  ```
    python test.py --cfg cfgs/${PATH_TO_CONFIG_FILE} --extra_tag {PREFERRED_NAME} --ckpt {PATH_TO_CKPT}
  ```
