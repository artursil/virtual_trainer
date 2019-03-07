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