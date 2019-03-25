"""
@author: Ahmad Kurdi
Model classes for Virtual Trainer portfolio project.
DSR portfolio project with Artur Silicki
Modifies Temporal CNN models from VideoPose3D + Quaternion utilities from Quaternet
"""

import torch
import torch.nn as nn
from VideoPose3D.common.model import TemporalModelOptimized1f, TemporalModel  

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def rotate_seq(seq, vects):

    # calculate euler angles
    
    e = (vects.permute(1,0)/ torch.norm(vects,dim=1)).permute(1,0)
    x = torch.mul(e[:, 0],torch.tensor(-1))
    y = torch.mul(e[:, 1],torch.tensor(-1))
    z = torch.mul(e[:, 2],torch.tensor(-1))

    # convert to quaternion order xyz
    rx = torch.stack((torch.cos(x/2), torch.sin(x/2), torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    ry = torch.stack((torch.cos(y/2), torch.zeros_like(y), torch.sin(y/2), torch.zeros_like(y)), dim=1)
    rz = torch.stack((torch.cos(z/2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z/2)), dim=1)
    q = qmul(rx,ry)
    q = torch.mul(qmul(q,rz),torch.tensor(1))

    # apply sequentially for each item in batch
    for i in range(seq.shape[0]):
        shp_ = torch.tensor(seq[i].shape)
        shp_[-1] = 4
        q_ = q[i].repeat(tuple(shp_[:-1])).reshape(tuple(shp_))
        seq[i] = qrot(q_, seq[i] )
    return seq

class SplitModel(nn.Module):
    """
    split output
    """
    def __init__(self, class_model):
        super().__init__()
        self.class_model = class_model
    def forward(self,x):
        pred = self.class_model(x).permute(0,2,1)
        embed = x.permute(0,2,1) # conv produces 1*128 , want flat 128
        return embed, pred

class SplitModel2(nn.Module):
    """
    split output
    """
    def __init__(self, class_model):
        super().__init__()
        self.class_model = class_model
        self.embed_layer = nn.Conv1d(128,64,1)
    def forward(self,x):
        pred = self.class_model(x).permute(0,2,1)
        embed = x.permute(0,2,1) # conv produces 1*128 , want flat 128
        return embed, pred

class StandardiseKeypoints(nn.Module):
    """
    transformer for standardising 3D keypoints
    """
    def __init__(self, centering=True, orientate=False):
        super().__init__()
        self.centering = centering
        self.orientate = orientate
        self.kp_buff = None
    
    def rotate_(self,x):
        direction_hip = x[:,0,4,:]
        direction_spine = x[:,0,7,:]
        x = rotate_seq(x,direction_hip) # orientate hips
        x = rotate_seq(x,direction_spine) # orientate spine
        return x

    def get_3dkeys(self):
        return self.kp_buff
        
    def forward(self, x):
        if self.centering:
            x-= x[:,0,0,:].unsqueeze(1).unsqueeze(1)
        if self.orientate:
            x = rotate_(x)
        self.kp_buff = x.detach().cpu().numpy()
        return x

    def get_kp(self):
        return self.kp_buff


class HeadlessNet(nn.Module):
    """
    Headless network - legacy. Do not use
    """
    def __init__(self, class_model, pretrained_weights):
        super().__init__()
        class_model.load_state_dict(pretrained_weights['model_state_dict'])
        class_model.top_model.shrink = HeadlessModule()
        self.embed_model = class_model
    def forward(self,x):
        x = self.embed_model(x)
        return x

class HeadlessNet2(nn.Module):
    """
    Headless network
    """
    def __init__(self, class_model):
        super().__init__()
        class_model.shrink = HeadlessModule()
        self.embed_model = class_model
    def forward(self,x):
        x = self.embed_model(x)
        return x

class RankingEmbedder(nn.Module):
    def __init__(self,input_embed,embed_lens):
        super().__init__()
        modules = []
        for i,layer in enumerate(embed_lens):
            modules.append(nn.Linear(input_embed,layer))
            input_embed = layer
            if i != len(embed_lens) - 1:
                modules.append(nn.ReLU())
        self.linears = nn.ModuleList(modules)
    def forward(self,x):        
        x= x.permute(0,2,1).squeeze()
        for module in self.linears:
            x = module(x)
        return x
        

class SiameseNet(HeadlessNet2):
    """
    Siamese network
    """
    def forward(self, x1, x2):
        x1 = self.embed_model(x1)
        x2 = self.embed_model(x2)
        return x1, x2


class TripletNet(HeadlessNet2):
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

class BaselineWithkeypoints(nn.Module):
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
        keypoints = x.detach().cpu().numpy() 
        x-= x[:,0,0,:].unsqueeze(1).unsqueeze(1)
        x = self.top_model(x)
        return x, keypoints        

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