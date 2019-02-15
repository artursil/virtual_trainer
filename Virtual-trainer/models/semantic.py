"""
@author: Ahmad Kurdi
Model classes for Virtual Trainer portfolio project.
DSR portfolio project with Artur Silicki
Modifies Temporal CNN models from VideoPose3D to allow transfer learning
"""

import torch
import torch.nn as nn
from VideoPose3D.common.model import TemporalModelOptimized1f, TemporalModel  


class ModdedStridedModel(TemporalModelOptimized1f):
    """
    Strided TCN model. Optimised for training
     
    Arguments:
    num_joints_in --  Number of input keypoints (as in pretrained model)
    in_features -- Number of features for each keypoint (as in pretrained model)
    num_joints_out -- Number of output keypoints (as in pretrained model)
    filter_widths -- List containing filter widths per block (as in pretrained model)
    channels -- Number of filter channels (as in pretrained model)
    causal -- Boolean to use causal convolutions (for realtime)
    """
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        super().__init__(num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels)

    # overload forward block to remove trailing permute
    def forward(self,x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        # flatten input
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x)))) 
        
        # iterate through blocks
        for i in range(len(self.pad) - 1):
            # don't take a residual if last block
            res = 0 if i == (len(self.pad) - 2) else x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        x = self.shrink(x) # Classifier unit
        return x

class ModdedTemporalModel(TemporalModel):
    """
    Reference TCN model. General model for evaluation and training
     
    Arguments:
    num_joints_in --  Number of input keypoints (as in pretrained model)
    in_features -- Number of features for each keypoint (as in pretrained model)
    num_joints_out -- Number of output keypoints (as in pretrained model)
    filter_widths -- List containing filter widths per block (as in pretrained model)
    channels -- Number of filter channels (as in pretrained model)
    causal -- Boolean to use causal convolutions (for realtime)
    """
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels, dense)

    # overload forward block to remove trailing permute
    def forward(self,x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            # don't take a residual if last block
            res = 0 if i == (len(self.pad) - 2) else x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        x = self.shrink(x)
        return x