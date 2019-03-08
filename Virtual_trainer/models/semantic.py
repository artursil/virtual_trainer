"""
@author: Ahmad Kurdi
Model classes for Virtual Trainer portfolio project.
DSR portfolio project with Artur Silicki
Modifies Temporal CNN models from VideoPose3D
"""

import torch
import torch.nn as nn
from VideoPose3D.common.model import TemporalModelOptimized1f, TemporalModel  


class HeadlessNet(nn.Module):
    """
    Headless network
    """
    def __init__(self, class_model, pretrained_weights):
        super().__init__()
        class_model.load_state_dict(pretrained_weights['model_state_dict'])
        class_model.top_model.shrink = HeadlessModule()
        self.embed_model = class_model
    def forward(self,x):
        x = self.embed_model(x)
        return x



class SiameseNet(HeadlessNet):
    """
    Siamese network
    """
    def forward(self, x1, x2):
        x1 = self.embed_model(x1)
        x2 = self.embed_model(x2)
        return x1, x2


class TripletNet(HeadlessNet):
    """
    Triplet network
    """
    def forward(self, x1, x2, x3):
        x1 = self.embed_model(x1)
        x2 = self.embed_model(x2)
        x3 = self.embed_model(x3)
        return x1, x2, x3


class HeadlessModule(nn.Module):
    """
    Swap-in for last layer to make a headless model
    """
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class NaiveStridedModel(nn.Module):
    """
    Baseline architecture to test out: Chain temporal models of same size: 
    (2d keypoints)--> [VP3D] >--(3d kypoints)--> [ModdedModel] >--(classes)
    Strided version for training
    """
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths,
                     pretrained_weights, embedding_len, classes, causal, dropout, channels, loadBase=True):
        super().__init__()
        self.base_model = TemporalModel(num_joints_in, in_features, num_joints_out, filter_widths,
                            causal=causal, dropout=dropout, channels=channels)
        if loadBase:
            self.base_model.load_state_dict(pretrained_weights['model_pos'])
        self.top_model = ModdedStridedModel(num_joints_out, 3, num_joints_out, filter_widths,
                                        causal=causal, dropout=dropout, channels=embedding_len, skip_res=False)
        self.top_model.shrink = nn.Conv1d( embedding_len, classes, 1)

    def forward(self, x):
        x = self.base_model(x)
        x-= x[:,0,0,:].unsqueeze(1).unsqueeze(1)
        x = self.top_model(x)
        return x

class NaiveBaselineModel(nn.Module):
    """
    Baseline architecture to test out: Chain temporal models of same size: 
    (2d keypoints)--> [VP3D] >--(3d kypoints)--> [ModdedModel] >--(classes)
    Reference version for running
    """
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths,
                     pretrained_weights, embedding_len, classes, causal, dropout, channels, loadBase=True):
        super().__init__()
        self.base_model = TemporalModel(num_joints_in, in_features, num_joints_out, filter_widths,
                            causal=causal, dropout=dropout, channels=channels)
        if loadBase:
            self.base_model.load_state_dict(pretrained_weights['model_pos'])
        self.top_model = ModdedTemporalModel(num_joints_out, 3, num_joints_out, filter_widths,
                                        causal=causal, dropout=dropout, channels=embedding_len, skip_res=False)
        self.top_model.shrink = nn.Conv1d( embedding_len, classes, 1)

    def forward(self, x):
        x = self.base_model(x)
        x-= x[:,0,0,:].unsqueeze(1).unsqueeze(1)
        x = self.top_model(x)
        return x        


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
                 filter_widths, causal=False, dropout=0.25, channels=1024, skip_res=True):
        self.skip_res = skip_res
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
            res = 0 if (i == (len(self.pad) - 2) and self.skip_res) else x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
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
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False, skip_res=True):
        self.skip_res = skip_res
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
            res = 0 if (self.skip_res and i == (len(self.pad) - 2)) else x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        x = self.shrink(x)
        return x