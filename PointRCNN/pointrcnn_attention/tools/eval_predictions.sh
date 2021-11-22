#!/bin/bash

cd ../../evaluate
for ((counter=2; counter<=70; counter = counter +2))


do

python evaluate.py --result_path ../pointrcnn/output/$1/val/final_result/data/  --dataset_path /media/HDD/vishwa/detection3d/st3d_v3/3D_adapt_auto_driving/pointrcnn/multi_data/$2/KITTI/object --metric  old
echo "Evaluated epoch $counter"


done >> $1.txt