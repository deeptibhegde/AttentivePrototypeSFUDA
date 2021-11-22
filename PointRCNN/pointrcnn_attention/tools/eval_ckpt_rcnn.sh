#!/bin/bash

echo $1
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt $1 --dataset $2 --output_dir ../output/$3  --batch_size 4  --eval_mode rcnn

cd /media/HDD/vishwa/detection3d/st3d_v3/3D_adapt_auto_driving/evaluate

python evaluate.py --result_path /media/HDD/vishwa/detection3d/st3d_v3/3D_adapt_auto_driving/pointrcnn/output/$3/eval/$4/val/final_result/data  --dataset_path /media/HDD/vishwa/detection3d/3D_adapt_auto_driving/pointrcnn/multi_data/$2/KITTI/object/   --metric old