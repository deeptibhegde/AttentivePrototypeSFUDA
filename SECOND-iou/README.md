# Implementation of Attentive Prototypes for SFUDA of [SECOND-iou](https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/kitti_models/second_iou.yaml) 

Format of configuration files from [ST3D](https://github.com/CVMI-Lab/ST3D/tree/master/tools/cfgs), based on the codebase from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md)

## Setup

This requires a specific environment setup. Make sure you have the [dependencies required](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) for SECOND-iou.

You will need 

 - Linux 16.04/14.04
 - Python 3.6
 - Cuda 10.2
 - [PyTorch 1.5](https://pytorch.org/get-started/previous-versions/#linux-and-windows-10)



1. Create a new conda environment 

   ```
   conda create -n secondiou python=3.6 -y
   
   conda activate secondiou
   ```


2. Install Pytorch 1.5

   ```
   conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
   ```
   
   
3. Build the [`spconv`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) library.

   

  
4. Build SECOND-iou

  ```
  cd pcdet
 ```

  ```
  python setup.py develop
  ```


5. Install dataset dependencies

```
pip install nuscenes-devkit==1.0.5
pip3 install waymo-open-dataset-tf-2-0-0==1.2.0 --user
```

# Datasets

1. Download the relevant datasets: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) , [Waymo](https://waymo.com/intl/en_us/dataset-download-terms/) , [nuScenes](https://www.nuscenes.org/download) and organize them according to the common instructions.

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


# Training

  `to be updated`
  
 # Evaluation
 
 1. Download the pretrained models.
 

2.  Evaluate a single checkpoint

```
python test.py --cfg cfgs/${PATH_TO_CONFIG_FILE} --batch_size 16   --extra_tag ${NAME} --ckpt ${PATH_TO_CKPT}
```
  
