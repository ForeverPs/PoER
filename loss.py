import torch
import torch.nn as nn
import torch.nn.functional as F


def get_dist(la, type):
    if type == 'l2':
        la = F.normalize(la)
        # batch x batch
        l = torch.sqrt(torch.sum(torch.square(la[:, None, :] - la), dim=-1) + 1e-20) # l2 distance from 0 to 1
        
    elif type == 'l1':
        la = F.normalize(la)
        # batch x batch
        l = torch.sum(torch.abs(la[:, None, :] - la), dim=-1) # l1 distance from 0 to 1

    elif type == 'cos':
        la = F.normalize(la)  # l2 norm equals to 1
        l = -(1.0 + la.mm(la.t())) / 2.  # cos distance from -1 to 0
    return l


def energy_ranking(feats, label, domain_label, type='l2', beta=1.0, margin=0., eps=1e-10):
    # feats : list of feat, batch x feat_dim
    # label : batch x 1
    # domain_label: batch x 1
    label = label.reshape(feats[0].shape[0], 1)
    domain_label = domain_label.reshape(feats[0].shape[0], 1)

    same_label_index = (label == label.t()).float()
    same_domain_index = (domain_label == domain_label.t()).float()

    # identical label and identical domain
    most_near_index = same_label_index * same_domain_index
    # identical label and different domain
    second_near_index = same_label_index * (1 - same_domain_index)
    # different label and same domain
    third_near_index = (1 - same_label_index) * same_domain_index
    # different label and different domain
    most_far_index = (1 - same_label_index) * (1 - same_domain_index)

    rank = nn.MarginRankingLoss(margin=margin)

    distance_index = [most_near_index, second_near_index, third_near_index, most_far_index]
    distance, loss = [get_dist(feat, type) for feat in feats], 0
    target = torch.tensor([-1]).reshape(1, 1).to(feats[0].device)
    for i, dist in enumerate(distance):
        pair_potential = torch.exp(beta * dist) - 1.0
        # domain-label ranking
        if i <= 2:
            rank_dist = [torch.sum(pair_potential * index) / torch.sum(index) for index in distance_index if torch.sum(index)]
            if len(rank_dist) >= 2:
                for j in range(len(rank_dist) - 1):
                    loss = loss + rank(rank_dist[j].reshape(1, 1), rank_dist[j + 1].reshape(1, 1), target)
        else:
            same_label_dist = torch.sum(pair_potential * same_label_index) / (torch.sum(same_label_index) + eps)
            diff_label_dist = torch.sum(pair_potential * (1 - same_label_index)) / (torch.sum(1.0 - same_label_index) + eps)
            loss = loss + (same_label_dist - diff_label_dist).exp()
    return loss
