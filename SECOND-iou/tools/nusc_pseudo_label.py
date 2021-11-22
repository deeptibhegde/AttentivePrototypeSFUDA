import pickle 
import numpy as np

with open('/media/labuser/HDD-2TB/deepti/ST3D_attention/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', 'rb') as ff:
    data_nusc = pickle.load(ff)


with open('/media/labuser/HDD-2TB/deepti/ST3D_attention/output/nuscenes_models/secondiou_car_oracle/waymo_nusc_pseudo/eval/epoch_4/val/default/result.pkl', 'rb') as f:
    data_pseudo = pickle.load(f)

# print(len(data_pseudo),len(data_nusc))

for sample in data_pseudo:
    for item in data_nusc:
        if sample['frame_id'] in item['lidar_path']:
            item['gt_boxes'] = sample['boxes_lidar']
            # print(type(item['gt_names']))
            item['gt_names'] = np.array(['car']*len(sample['boxes_lidar']))
            # print(item['gt_names'])

            print(sample['frame_id'])




dbfile = open('/media/labuser/HDD-2TB/deepti/ST3D_attention/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train_it1.pkl', 'ab')
      
# source, destination
pickle.dump(data_nusc, dbfile)                     
dbfile.close()