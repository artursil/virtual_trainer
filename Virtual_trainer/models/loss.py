import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations , product

def dist_mat(embeddings):
    # Reconstruct distance matrix because torch.pdist gives back a condensed flattened vector
    
    n_samples = embeddings.squeeze_().shape[0]
    mat = torch.zeros(n_samples,n_samples)
    if torch.cuda.is_available():
        mat = mat.cuda()
    dists = F.pdist(embeddings)
    s_ = 0
    for i , n in enumerate(reversed(range(1,n_samples))):
        mat[i,i+1:] = dists[s_:s_+n]
        s_ += n
    return mat


class ContrastiveLoss(nn.MarginRankingLoss):
    # modified contrastive loss function that scales loss by range of rankings

    def __init__(self, margin=0., size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin
        self.loss_tuples = []

    def forward(self, x1, x2, difs):
        mx_dif = torch.max(difs)
        dists = F.pairwise_distance(x1,x2) - difs
        good_form = torch.mul((mx_dif - difs)/mx_dif, dists.pow(2))
        bad_form = torch.mul(difs/mx_dif, F.relu(dists-self.margin).pow(2))
        losses = (good_form  + bad_form ) / 2
        self.loss_tuples.append( list(zip(dists.detach().cpu().numpy() , difs.detach().cpu().numpy())) )
        return torch.mean(losses)

    def get_tuples(self):
        return self.loss_tuples



class CustomRankingLoss(nn.MarginRankingLoss):
    #  loss function that for each exercise selects hard pairs of positives and negatives (adjusted for expected distance) and finds MSE of their L2 distances

    def __init__(self, margin=0., size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin
        self.pairings = []

    def forward(self, embeddings, classes, rankings):
        loss = self.forward_(embeddings, classes, rankings)
        return loss

    
    def forward_(self, embeddings, classes, rankings):
        # if rankings.dtype==torch.int64:
        #     rankings = rankings.unsqueeze(0)
        # if classes.dtype==torch.int64:
        #     classes = classes.unsqueeze(0)
        my_zero = torch.zeros(1)
        my_one = torch.ones(1)
        my_empty = torch.empty((0))
        if torch.cuda.is_available():
            my_empty = my_empty.cuda()
            my_one = my_one.cuda()
            my_zero = my_zero.cuda()
        distances = my_empty
        pairings = []   

        top_mark = torch.max(rankings) # get the top rating (should be 9)
        for ex_class in torch.unique(classes):
            ex_mask = torch.nonzero(torch.where(classes == ex_class,my_one,my_zero))[:,0]
            pos_mask = torch.nonzero(torch.where(rankings[ex_mask] == top_mark,my_one,my_zero))[:,0]
            if len(pos_mask)<2:
                continue
            neg_mask = torch.nonzero(torch.where(rankings[ex_mask] == top_mark,my_zero,my_one))[:,0]
            pos_dists = torch.max(dist_mat(embeddings[ex_mask[pos_mask]]),dim=1)# find hard positives
            # save pairings and distances for positives
            pairings.append(np.stack((ex_mask[pos_mask].detach().cpu().numpy(),ex_mask[pos_dists[1]].detach().cpu().numpy() ), axis=0))
            distances = torch.cat((distances,pos_dists[0]), dim=0)

            if len(neg_mask)<1:
                continue
            for positive in pos_mask:
                emb_ = embeddings[ex_mask[positive]].repeat(neg_mask.shape[0],1)
                exp_dist = top_mark - rankings[ex_mask[neg_mask]] # expected distance
                hard_neg = torch.max(torch.abs(F.pairwise_distance(emb_, embeddings[ex_mask[neg_mask]]) - exp_dist.to(torch.float32)),dim=0)
                # save pairing and distance of hard negative
                pairings += [ex_mask[positive].detach().cpu().numpy(),ex_mask[torch.max(hard_neg[1])].detach().cpu().numpy()]
                distances = torch.cat( (distances, torch.tensor(torch.max(hard_neg[0])).unsqueeze(0)), dim=0)
        loss = torch.mean(F.relu(distances-self.margin).pow(2))
        self.pairings = pairings
        return loss

    def get_pairings(self):
        return self.pairings

class CombinedLoss(CustomRankingLoss):

    def __init__(self, cl_loss, weighting, supress_cl=6, margin=0., size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin
        self.pairings = []
        self.weighting = weighting
        self.cl_loss = cl_loss
        self.supress_cl = supress_cl

    def forward(self, embeddings, preds, classes, rankings):
        weighting = torch.tensor(self.weighting)
        preds = preds.squeeze()
        embeddings = embeddings.squeeze()
        classes = classes.squeeze()
        # take classification loss
        classloss = self.cl_loss(preds,classes)
        
        # supress control group (non exercise)
        my_zero = torch.zeros(1)
        my_one = torch.ones(1)
        if torch.cuda.is_available():
            my_one = my_one.cuda()
            my_zero = my_zero.cuda()
        mask = torch.nonzero(torch.where(classes == self.supress_cl,my_zero,my_one))[:,0]

        import pdb
        #pdb.set_trace()
        # take ranking loss
        rankloss = self.forward_(embeddings[mask], classes[mask], rankings[mask])

        print(f"Unweighted losses: Classification C/E {classloss} , Ranking MSE {rankloss}")
        # scale by weighting
        if torch.cuda.is_available():
            weighting = weighting.cuda()
        loss = torch.mul(rankloss, (1-weighting)) + torch.mul(classloss, weighting) 

        return loss
    
    def set_weighting(self, weighting):
        assert weighting < 1
        self.weighting = weighting
