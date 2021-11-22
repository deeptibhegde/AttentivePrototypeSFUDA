import os


val = "/media/HDD/vishwa/detection3d/st3d_v3/3D_adapt_auto_driving/pointrcnn/multi_data/nusc/KITTI/ImageSets/train.txt"

val_f = open(val,'r+')

samples = val_f.readlines()


folder = os.listdir("/media/HDD/vishwa/detection3d/st3d_v3/3D_adapt_auto_driving/pointrcnn/multi_data/nusc/KITTI/ImageSets/training/velodyne/")


train_out = "/media/HDD/vishwa/detection3d/st3d_v3/3D_adapt_auto_driving/pointrcnn/multi_data/nusc/KITTI/ImageSets/train_out.txt"


out_f = open(train_out,'w+')
# import pdb; pdb.set_trace()
for sample in samples:
    if sample[:-1] + '.bin' in folder:
        out_f.write(sample)

out_f.close()