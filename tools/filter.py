import os 
import numpy as np
from skimage import io

path = "/data/datasets/kitti_all/training/label_2_waymo_ped_1"
# files = os.path.listdir(path)

val_in = open("/data/datasets/kitti_all/ImageSets/val.txt")
train_in = open("/data/datasets/kitti_all/ImageSets/train.txt")



val_out = open("/data/datasets/kitti_all/ImageSets/val_f.txt",'w+')
train_out = open("/data/datasets/kitti_all/ImageSets/train_f.txt",'w+')

# val = open("/data/datasets/kitti_all/ImageSets/val.txt")
# train = open("/data/datasets/kitti_all/ImageSets/train.txt")

# train_files = train.readlines()
# val_files = val.readlines()


train_files = train_in.readlines()
val_files = val_in.readlines()

for f in train_files:
    # img_file = "/data/datasets/kitti_all/training/image_2/" + f[:-4] + '.png'
    # try:
        
    #     np.array(io.imread(img_file).shape[:2], dtype=np.int32)
    # except:
    #     print("ERROR")
    #     continue
        

    fo = open(os.path.join(path,f[:-1] + '.txt'))

    lines = fo.readlines()
    if len(lines)> 0:
        train_out.write(f)

for f in val_files:

    # img_file = "/data/datasets/kitti_all/training/image_2/" + f[:-4] + '.png'
    # try:
        
    #     np.array(io.imread(img_file).shape[:2], dtype=np.int32)
    # except:
    #     print("ERROR")
    #     continue

    try:
        fo = open(os.path.join(path,f[:-1] + '.txt'))
    except:
        continue
    lines = fo.readlines()

    if len(lines)> 0:
        val_out.write(f)