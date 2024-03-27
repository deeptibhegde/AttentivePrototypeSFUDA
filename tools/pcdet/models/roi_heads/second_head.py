from functools import partial

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, loss_utils

import numpy as np

import os

from .vit import Transformer
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
class SECONDHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = self.model_cfg.ROI_GRID_POOL.IN_CHANNEL * GRID_SIZE * GRID_SIZE

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.iou_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=1, fc_list=self.model_cfg.IOU_FC
        )
        self.init_weights(weight_init='xavier')

        if torch.__version__ >= '1.3':
            self.affine_grid = partial(F.affine_grid, align_corners=True)
            self.grid_sample = partial(F.grid_sample, align_corners=True)
        else:
            self.affine_grid = F.affine_grid
            self.grid_sample = F.grid_sample

        if self.model_cfg.get('PROTO', False):
            self.dropout = nn.Dropout(0.1)
            self.transformer_module = Transformer(dim=256, depth=4, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1)

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                spatial_features_2d: (B, C, H, W)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d'].detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

        dataset_cfg = batch_dict['dataset_cfg']
        min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
        min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
        voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
        voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
        down_sample_ratio = self.model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO

        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            # Map global boxes coordinates to feature map coordinates
            x1 = (rois[b_id, :, 0] - rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            x2 = (rois[b_id, :, 0] + rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            y1 = (rois[b_id, :, 1] - rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            y2 = (rois[b_id, :, 1] + rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

            angle, _ = common_utils.check_numpy_to_torch(rois[b_id, :, 6])

            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            grid = self.affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size))
            )

            pooled_features = self.grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid
            )

            pooled_features_list.append(pooled_features)

        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)

        return pooled_features

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        # prototype = None
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            prototype = batch_dict['prototype']
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, C, 7, 7)
        batch_size_rcnn = pooled_features.shape[0]

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))

        # if os.path.isfile(prototype_file) and self.model_cfg.get('PROTO', False):
        #     prototype = np.load(prototype_file)
        #     batch_dict['prototype'] = prototype
        

        # and self.model_cfg.get('ATT', False)
        if self.model_cfg.get('PROTO', False) and self.training:
            mask = targets_dict['gt_of_rois_src'][:,:,-1] > 0
            ind = np.where(targets_dict['gt_of_rois_src'][:,:,-1].detach().cpu().numpy() > 0)
            mask = mask.view(-1)
            region_features = shared_features[mask].unsqueeze(0).squeeze(-1)

            # import pdb; pdb.set_trace()        
            
            
            if prototype is not None and self.model_cfg.get('ATT', False):
                

                alpha = self.model_cfg.ALPHA
                
                # import pdb; pdb.set_trace()        
                region_features = self.dropout(region_features)
                
                region_features_t = self.transformer_module(region_features.squeeze(-1)).squeeze(0)


                # import pdb; pdb.set_trace() 

                if self.model_cfg.get('UPDATE_RF', False):
                    shared_features[mask] = region_features_t.unsqueeze(-1)


                rcnn_iou = self.iou_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*N, 1)

                if self.model_cfg.get('ENTROPY_WEIGH', False):
                    rcnn_iou_prob = torch.nn.functional.sigmoid(rcnn_iou)
                    entropy = torch.squeeze(-1*((rcnn_iou_prob)*torch.log(rcnn_iou_prob) + (1 - rcnn_iou_prob)*torch.log(1 - rcnn_iou_prob)))

                    entropy_weight = 1 - entropy

                    current_prototype = torch.transpose((torch.transpose(region_features_t,0,1)*entropy_weight),0,1).mean(dim=0)
                else:
                    current_prototype = torch.squeeze(torch.mean(region_features_t,dim=0,keepdim=True),dim=-1)
                # current_prototype = torch.unsqueeze(current_prototype,dim=0)
                

                

                # import pdb; pdb.set_trace()
                    
                final_prototype = (alpha*torch.tensor(batch_dict['prototype']).cuda() + (1-alpha)*current_prototype)

                final_prototype /= final_prototype.norm(dim=-1, keepdim=True)
                region_features_t /= region_features_t.norm(dim=-1, keepdim=True)

                feature_sim = region_features_t @ final_prototype.squeeze(0).t()



            else:
                rcnn_iou = self.iou_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
                final_prototype = torch.squeeze(torch.mean(region_features,dim=1,keepdim=True),dim=-1)

                final_prototype /= final_prototype.norm(dim=-1, keepdim=True)
                region_features /= region_features.norm(dim=-1, keepdim=True)

                feature_sim = region_features.squeeze() @ final_prototype.squeeze(0).t()
                
            np.save(os.path.join(batch_dict['proto_dir'],'prototype_epoch_%d.npy'%batch_dict['epoch']),final_prototype.detach().cpu().numpy())

            # import pdb; pdb.set_trace()
            # final_prototype = final_prototype.squeeze(0)
            # region_features = region_features.squeeze()
            
            


            
            
            sim_weights = torch.nn.functional.softmax(feature_sim,dim=0).detach()

            
        else:
            rcnn_iou = self.iou_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)    


        

        if not self.training:
            batch_dict['batch_cls_preds'] = rcnn_iou.view(batch_dict['batch_size'], -1, rcnn_iou.shape[-1])
            batch_dict['batch_box_preds'] = batch_dict['rois']
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_iou'] = rcnn_iou

            self.forward_ret_dict = targets_dict
            if self.model_cfg.get('PROTO', False):
                self.forward_ret_dict['sim_weights'] = sim_weights
                self.forward_ret_dict['ind'] = ind

                self.forward_ret_dict['prototype'] = final_prototype

        

        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_iou_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        tb_dict['rcnn_loss'] = rcnn_loss.item()

        # if 'prototype' in self.forward_ret_dict.keys():
        #     tb_dict['prototype'] = self.forward_ret_dict['prototype']
            
        
        return rcnn_loss, tb_dict

    def get_box_iou_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_iou = forward_ret_dict['rcnn_iou']
        rcnn_iou_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_iou_flat = rcnn_iou.view(-1)
        if loss_cfgs.IOU_LOSS == 'BinaryCrossEntropy':
            batch_loss_iou = nn.functional.binary_cross_entropy_with_logits(
                rcnn_iou_flat,
                rcnn_iou_labels.float(), reduction='none'
            )
        elif loss_cfgs.IOU_LOSS == 'L2':
            batch_loss_iou = nn.functional.mse_loss(rcnn_iou_flat, rcnn_iou_labels, reduction='none')
        elif loss_cfgs.IOU_LOSS == 'smoothL1':
            diff = rcnn_iou_flat - rcnn_iou_labels
            batch_loss_iou = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(diff, 1.0 / 9.0)
        elif loss_cfgs.IOU_LOSS == 'focalbce':
            batch_loss_iou = loss_utils.sigmoid_focal_cls_loss(rcnn_iou_flat, rcnn_iou_labels)
        else:
            raise NotImplementedError

        # import pdb; pdb.set_trace()
        iou_valid_mask = (rcnn_iou_labels >= 0)
        # import pdb; pdb.set_trace()
        if self.model_cfg.get('PROTO', False):
            rcnn_loss_iou = (batch_loss_iou[self.forward_ret_dict['ind'][0]] * self.forward_ret_dict['sim_weights'].squeeze()).sum()#/ torch.clamp(iou_valid_mask.sum(), min=1.0)
        else:
            rcnn_loss_iou = (batch_loss_iou * iou_valid_mask).sum()/ torch.clamp(iou_valid_mask.sum(), min=1.0)

        rcnn_loss_iou = rcnn_loss_iou * loss_cfgs.LOSS_WEIGHTS['rcnn_iou_weight']
        tb_dict = {'rcnn_loss_iou': rcnn_loss_iou.item()}
        return rcnn_loss_iou, tb_dict
