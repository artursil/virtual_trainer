import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

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

def select_triplets(embeddings,classes,margin):
    # find hard negatives (from higher classes aka higher ratings) for combinations of positive pairs
    dists = dist_mat(embeddings)
    triplets = []
    for label in torch.unique(classes,sorted=True):
        pos_ind = torch.nonzero(torch.where(classes == label,torch.ones(1),torch.zeros(1)))[:,0]
        neg_ind = torch.LongTensor(torch.nonzero(torch.where(classes > label,torch.ones(1),torch.zeros(1)))[:,0])
        if (len(pos_ind) < 2 or len(neg_ind) == 0):
            continue
        pos_pairs = torch.LongTensor(list(combinations(pos_ind,2)))
        pos_dists = dists[pos_pair[:, 0], pos_pair[:, 1]]
        for pos_pair, pos_dist in zip(pos_pairs,pos_dists):
            losses = margin + pos_dist - dists[pos_pair[0],neg_ind]
            hard_neg = torch.argmax(losses)
            if hard_neg is not None:
                    triplets.append([pos_pair[0], pos_pair[1], neg_ind[hard_neg]])

    if len(triplets) == 0:
        triplets.append([pos_pair[0], pos_pair[1], neg_ind[0]])

    return torch.LongTensor(triplets)


class OnlineTriplet(nn.TripletMarginLoss):
    def forward(self, embeddings, classes):
        # finds hard triplets from batch. Assumes balanced batch of 3 classes (consecutive ratings)
        anc, pos, neg = select_triplets(embeddings,classes,self.margin)
        return F.triplet_margin_loss(embeddings[anc], embeddings[pos], embeddings[neg], margin=self.margin, p=self.p,
                                    eps=self.eps, swap=self.swap, reduction=self.reduction)



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
