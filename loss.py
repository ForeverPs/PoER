import torch
import torch.nn as nn
from utils import sinkhorn
import torch.nn.functional as F


def recon_loss(recon_x, input_x):
    mse_loss = F.binary_cross_entropy(recon_x, input_x)
    return mse_loss


def energy_ranking(la, label, beta=.1, type='cos', log_space=False, margin=0.3):
    # la : batch x feat_dim
    # label : batch x 1
    label = label.reshape(la.shape[0], 1)
    if type == 'emd':
        b, k = la.shape
        la = F.normalize(la)
        la1 = la.unsqueeze(0)  # 1 x batch x k
        la2 = la.unsqueeze(1)  # batch x 1 x k
        la1 = la1.repeat((b, 1, 1)).reshape(b * b, k)
        la2 = la2.repeat((1, b, 1)).reshape(b * b, k)
        c = (2 * torch.ones(k, k) - torch.eye(k)).repeat((b * b, 1, 1))
        l = sinkhorn(c, la1, la2, log_space=log_space)

    elif type == 'l2':
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

    # 0 for same categories, 1 for different categories
    target = 1 - (label == label.t()).float().reshape(1, -1)
    pair_potential = torch.exp(beta * l).reshape(1, -1)

    # same_category_energy = torch.sum((1 - target) * pair_potential) / (1 + torch.sum(1 - target))
    # diff_category_energy = torch.sum(target * pair_potential) / (1 + torch.sum(target))
    # loss = same_category_energy - diff_category_energy
    
    energy_diff = pair_potential - pair_potential.t()
    label_diff = torch.sign(target - target.t())  # b x b  0: no loss, -1: same-different, 1: different-same
    objective = -energy_diff * label_diff + margin 
    loss_value = torch.sum((objective + torch.abs(objective)) / 2)  # sum of positive value
    loss_num = torch.sum(torch.sign(objective + torch.abs(objective)))  # number of positive value
    loss = loss_value / (loss_num + 1e-10)
    return loss


# def energy_ranking(la, label, bits, beta=1., log_space=False):
#     # la : batch x length
#     # label : batch x 1
#     # c : batch x length x length

#     # normalize
#     b, k = la.shape
#     assert k % bits == 0
#     dim = k // bits
#     la1 = (la / torch.sum(la, dim=1).unsqueeze(-1)).unsqueeze(0)  # 1 x batch x k
#     la2 = (la / torch.sum(la, dim=1).unsqueeze(-1)).unsqueeze(1)  # batch x 1 x k
#     la1 = la1.repeat((b, 1, 1)).reshape(b * b, k)
#     la2 = la2.repeat((1, b, 1)).reshape(b * b, k)
#     tensor_list = [torch.ones((bits, bits)) for _ in range(dim)]
#     c = (2 * torch.ones((k, k)) - torch.block_diag(*tensor_list)).repeat((b * b, 1, 1))

#     l = sinkhorn(c, la1, la2, log_space=log_space)
#     # 0 for same categories, 1 for different categories
#     target = (label == label.t()).float().reshape(1, -1)
#     pair_potential = torch.exp(beta * l).reshape(1, -1)

#     # margin ranking
#     energy_diff = pair_potential - pair_potential.t()  # b x b
#     label_diff = torch.sign(target - target.t())  # b x b
#     objective = -energy_diff * label_diff
#     loss_value = torch.sum((objective + torch.abs(objective)) / 2)
#     loss_num = torch.sum(torch.sign(objective + torch.abs(objective)))
#     loss = loss_value / (loss_num + 1e-10)
#     return loss


if __name__ == '__main__':
    # la : batch_size x coding_dim
    # label : batch_size x 1
    # batch_size = 4
    # latent_dim = 2

    # la = torch.rand(batch_size, latent_dim).float()
    # la.requires_grad = True
    # label = torch.tensor([0, 0, 1, 1]).reshape(4, 1).long()
    # while True:
    #     loss = energy_ranking(la, label, beta=1., type='l2', log_space=False)
    #     print('loss', loss)
    #     if torch.isnan(loss):
    #         break
    #     # break
    #     loss.backward()
    #     la = torch.autograd.Variable(la - 0.1 * la.grad.data, requires_grad=True)
    #     print(loss, la, label)
    #     break

    x = torch.rand(10, 512)
    label = (10 * torch.rand(10)).long()
    loss = energy_ranking(x, label)
    print(loss)

