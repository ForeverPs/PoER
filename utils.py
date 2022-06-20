import torch
import numpy as np
import torch.nn as nn

def calculate_recon_error(recon, x):
    # recon : batch, channel, width, height
    # x : batch, channel, width, height
    dist = torch.mean(torch.square(recon - x), dim=[1, 2, 3])
    return dist
    

def calculate_cls_conf(cls):
    conf = nn.Softmax(-1)(cls)
    conf, _ = torch.max(conf, dim=-1)
    return conf


def calculate_pair_potential(train_feat, train_feat_label, ranking, num_classes=10, beta=.1):
    # train_feat : Nxdim
    # train_feat_label : N
    # ranking : Mxdim
    train_feat = train_feat.cpu().numpy()
    train_feat_label = train_feat_label.cpu().numpy()
    ranking = ranking.cpu().numpy()
    # cos dist: -1~0
    cos_dist = -(1 + np.matmul(ranking, train_feat.transpose())) / 2.0  # shape: MxN
    min_category_dist = list()
    for i in range(num_classes):
        index = np.where(train_feat_label == i)[0]
        mean_cos_dist = np.mean(cos_dist[:, index], axis=-1)
        min_category_dist.append(mean_cos_dist[..., np.newaxis])
    min_category_dist = np.concatenate(min_category_dist, axis=-1)
    min_category_dist = np.min(min_category_dist, axis=-1)
    pair_potential = np.exp(beta * min_category_dist)
    return torch.from_numpy(pair_potential)


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def sinkhorn_iterations(Z: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, iters: int) -> torch.Tensor:
    u, v = torch.ones_like(mu), torch.ones_like(nu)
    for _ in range(iters):
        u = mu / torch.einsum('bjk,bk->bj', [Z, v])
        v = nu / torch.einsum('bkj,bk->bj', [Z, u])
    return torch.einsum('bk,bkj,bj->bjk', [u, Z, v])


def sinkhorn(C, a, b, eps=2e-1, n_iter=10, log_space=True):
    """
    Args:
        a: tensor, normalized, note: no zero elements
        b: tensor, normalized, note: no zero elements
        C: cost Matrix [batch, n_dim, n_dim], note: no zero elements
    """
    P = torch.exp(-C/eps)
    if log_space:
        log_a = a.log()
        log_b = b.log()
        log_P = P.log()
    
        # solve the P
        log_P = log_sinkhorn_iterations(log_P, log_a, log_b, n_iter)
        P = torch.exp(log_P)
    else:
        P = sinkhorn_iterations(P, a, b, n_iter)
    return torch.sum(C * P, dim=[1, 2])


if __name__ == '__main__':
    train_feat = torch.rand(100, 32)
    train_feat_label = (10 * torch.rand(100)).long()
    ranking = torch.rand(60, 32)
    pair_potential = calculate_pair_potential(train_feat, train_feat_label, ranking, num_classes=10)
    print(pair_potential.shape)