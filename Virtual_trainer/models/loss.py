import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations , product

def dist_mat(embeddings):
    # Reconstruct distance matrix because torch.pdist gives back a condensed flattened vector
    n_samples = embeddings.shape[0]
    mat = torch.zeros(n_samples,n_samples)
    dists = F.pdist(embeddings)
    s_ = 0
    for i , n in enumerate(reversed(range(1,n_samples))):
        mat[i,i+1:] = dists[s_:s_+n]
        s_ += n
    return mat


class ContrastiveLoss(nn.MarginRankingLoss):
    # modified contrastive loss function that scales loss by range of rankings

    def __init__(self, dif_range, margin=0., size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin
        self.dif_range = dif_range
        self.loss_tuples = []

    def forward(self, x1, x2, difs):
        scaled_dif = difs / self.dif_range
        dists = F.pairwise_distance(x1,x2) / self.dif_range
        good_form = torch.mul((1 - scaled_dif), dists.pow(2))
        bad_form = torch.mul(scaled_dif, F.relu(torch.mul(difs,self.margin) - dists)).pow(2)
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
        self.pairings = torch.empty((0))

    def forward(self, embeddings, classes, rankings):
        distances = torch.empty((0))
        pairings = torch.empty((0))
        top_mark = torch.max(rankings) # get the top rating (should be 9)
        for ex_class in torch.unique(classes):
            ex_mask = torch.nonzero(torch.where(classes == ex_class,torch.ones(1),torch.zeros(1)))[:,0]
            pos_mask = torch.nonzero(torch.where(rankings[ex_mask] == top_mark,torch.ones(1),torch.zeros(1)))[:,0]
            neg_mask = torch.nonzero(torch.where(rankings[ex_mask] == top_mark,torch.zeros(1),torch.ones(1)))[:,0]
            pos_dists = torch.max(dist_mat(embeddings[ex_mask[pos_mask]]),dim=1) # find hard positives
            # save pairings and distances for positives
            pairings = torch.cat( (pairings , (torch.stack((ex_mask[pos_mask],ex_mask[pos_dists[1]]))) ), dim=0)
            distances = torch.cat((distances,pos_dists[0]), dim=0)
            for positive in pos_mask:
                emb_ = embeddings[ex_mask[positive]].repeat(neg_mask.shape[0],1)
                exp_dist = top_mark - rankings[ex_mask[neg_mask]] # expected distance
                hard_neg = torch.max(torch.abs(F.pairwise_distance(emb_, embeddings[ex_mask[neg_mask]]) - exp_dist),dim=0)
                # save pairing and distance of hard negative
                pairings = torch.cat( (pairings, torch.tensor([ex_mask[positive],ex_mask[hard_neg[1]]])) , dim=0)
                distances = torch.cat( (distances, torch.tensor([hard_neg[0]])), dim=0)
        loss = torch.mean(F.relu(distances-self.margin).pow(2))
        self.pairings = pairings
        return loss

    def get_pairings(self):
        return self.pairings


