import numpy as np
import torch.nn as nn
from .anchor_head_template import AnchorHeadTemplate
from .mhsa import MultiHeadSelfAttention
import torch
from .self_attention import SA_block
from .vit import Transformer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None


        self.dropout = nn.Dropout(0.1)
        self.transformer_module = Transformer(dim=256, depth=4, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1)
      


        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict,prototype):
        spatial_features_2d = data_dict['spatial_features_2d']




        ############ classification branch before SA ##################

        # if not self.training:
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['spatial_features_2d'] = spatial_features_2d

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        ######################################################################################

        ########################### prototype update and filter #################################
        if self.training:

            box_cls_labels = self.forward_ret_dict['box_cls_labels'].view(self.forward_ret_dict['box_cls_labels'].shape[0],-1,1)
            mask = torch.squeeze(box_cls_labels>0)
            ind = np.where(box_cls_labels.cpu()>0)

    
            region_features = spatial_features_2d.view(spatial_features_2d.shape[0],-1,box_cls_labels.shape[1])
            region_features = region_features.permute(0,2,1).contiguous()


            region_features_out = torch.unsqueeze(region_features[mask],dim=0) 

            alpha = 0.9999

            if prototype is not None:
                

                if region_features_out.shape[-1]>0:


                    x = region_features_out
                    b, n, _ = x.shape

                    x = self.dropout(x)
                    attentive_prototype = self.transformer_module(x)




                    attentive_prototype = torch.squeeze(attentive_prototype)


                #************************************************combine with entropy weights*************************************************#
              
                    cls_preds_reshape = cls_preds.view(self.forward_ret_dict['box_cls_labels'].shape)
                    cls_preds_prob = torch.nn.functional.sigmoid(cls_preds_reshape[mask])

                    cls_preds_prob_total = torch.nn.functional.sigmoid(cls_preds_reshape)

                    entropy = torch.squeeze(-1*((cls_preds_prob)*torch.log(cls_preds_prob) + (1 - cls_preds_prob)*torch.log(1 - cls_preds_prob)))



                    entropy_weight = 1 - entropy

                    current_prototype = torch.transpose((torch.transpose(attentive_prototype,0,1)*entropy_weight),0,1).mean(dim=0)

                    current_prototype = torch.unsqueeze(current_prototype,dim=0)

                #************************************************combine with average*********************************************************#
                # current_prototype =   torch.squeeze(torch.mean(attentive_prototype,dim=0,keepdim=True),dim=1)## #
                #*****************************************************************************************************************************#

                
                    final_prototype = (alpha*prototype.cuda().detach() + (1-alpha)*current_prototype) #torch.mean(torch.stack((prototype,current_prototype)),dim=0)
                else:
                    final_prototype = prototype
                    print('empty mask')
            else:
                final_prototype = torch.squeeze(torch.mean(region_features_out,dim=1,keepdim=True),dim=1)


            if region_features_out.shape[-1]>0:

                feature_sim = self.cos(torch.squeeze(region_features_out),final_prototype)
                sim_weights = torch.nn.functional.softmax(feature_sim).detach()

                cls_preds = cls_preds.view(cls_preds.shape[0],-1,1)

                self.forward_ret_dict['sim_weights'] = 10*sim_weights
                self.forward_ret_dict['ind'] = ind


            

            return data_dict,final_prototype
        
        return data_dict,None
