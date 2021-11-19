# Implementation for [PointRCNN](https://github.com/sshaoshuai/PointRCNN) 

Data processing steps from [this codebase](https://github.com/cxy1997/3D_adapt_auto_driving) by [Wang et al](https://arxiv.org/abs/2005.08139)

## Setup

This requires a different environment setup. Make sure you have the [dependencies required](https://github.com/sshaoshuai/PointRCNN#installation) for PointRCNN.

You will need 

 - Linux 16.04/14.04
 - Python 3.6
 - Cuda 9.0 or 10.0
 - [PyTorch 1.0](https://pytorch.org/get-started/previous-versions/#v100)



1. Create a new conda environment 

   ```
   conda create -n pointrcnn python=3.6 -y
   ```
  
   ```
   conda activate pointrcnn
   ```


2. Install Pytorch 1.0

   ```
   conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch
   ``` 
   (for Cuda 10.0) 

   ```
   conda install pytorch==1.0.0 torchvision==0.2.1 cuda90 -c pytorch
   ``` 
   (for Cuda 9.0)  
  
3. Build PointRCNN

  ```
  cd pointrcnn_attention
  ```

  ```
  bash build_and_install.sh
  ```
  
  
4. Install dependencies `tqdm`, `skimage`, `pyyaml`, `easydict`, `tensorboardX`

# Datasets

1. Please download the relevant datasets: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) , [Waymo](https://waymo.com/intl/en_us/dataset-download-terms/) , [nuScenes](https://www.nuscenes.org/download) and organize them according to the common instructions.

2. Convert the data to the correct format using [this codebase](https://github.com/cxy1997/3D_adapt_auto_driving#training-to-be-updated)


# Training

`To be updated`

# Evaluation

1. Download the pretrained models.

2. Run inference the model for a single checkpoint

   ```
   cd pointrcnn_attention/tools
   ```

   ```
   python eval_rcnn.py  --cfg_file cfgs/default.yaml --batch_size 16  --eval_mode rcnn  --ckpt ../models/{$CKPT_NAME}.pth  --dataset {$TARGET_DATASET}  --output_dir  ../output/$SOURCE_to_$TARGET/rcnn
   ```
   
    where `$TARGET_DATASET` is `kitti` or `nusc`
    
3. Generate evaluation scores for a single ckpt

   ```
   cd ../../evaluate
   ```
   
   ```
   python evaluate.py --result_path ../pointrcnn/output/$PATH_TO_PREDICTIONS  --dataset_path ../pointrcnn/multi_data/$TARGET_DATASET/KITTI/object --metric  old/new
   ```

    
    
 
4. Run inference the model for all checkpoints

   ```
   cd pointrcnn_attention/tools
   ```

   ```
   python eval_rcnn.py  --cfg_file cfgs/default.yaml --batch_size 16  --eval_mode rcnn --eval_all  --ckpt_dir ../output/$PATH_TO_CKPT_FOLDER/.  --dataset $TARGET_DATASET}  --output_dir  ../output/$SOURCE_to_$TARGET/rcnn
   ```
   
    where `$TARGET_DATASET` is `kitti` or `nusc`
    
5. Generate evaluation scores for all ckpt


    ```
    bash eval.sh $PATH_TO_EVAL_FOLDER $TARGET_DATASET
    ```
    
    
 
    
    




