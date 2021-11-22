import os
import numpy as np
import shutil


path = "/media/labuser/HDD-2TB/deepti/ST3D_attention/output/kitti_models/secondiou_oracle_ros/waymo_it4_loss/eval/epoch_37/val/default/final_result/data/"
dest = "/media/labuser/HDD-2TB/deepti/ST3D_attention/data/kitti/training/waymo_kitti_it5_0p8_loss/"

if not os.path.isdir(dest):
	os.mkdir(dest)

t = open("/media/labuser/HDD-2TB/deepti/ST3D_attention/data/kitti/ImageSets/train_waymo_kitti_it5_0p8_loss.txt",'w+')

it = os.listdir(path)

for file in it:
	f = open(path + file)

	if f.readlines():
		shutil.copy(path + file, dest + file)
		t.write(file[:-4] + '\n')


t.close()