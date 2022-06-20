import os
import tqdm
import torch
import shutil
import numpy as np
from data import *
import torch.nn as nn
from model import PoER
from torch.optim import Adam
from eval import get_metric, get_acc
from tensorboardX import SummaryWriter
from loss import recon_loss, energy_ranking
from torch.optim.lr_scheduler import StepLR
from utils import calculate_pair_potential, calculate_cls_conf, calculate_recon_error


device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')


def train(epochs, batch_size, lr):
    train_loader, val_loader = cifar10_data_loader(batch_size=batch_size)
    ood_loader = lsun_data_loader(batch_size=batch_size)

    # train_loader, val_loader = mnist_data_loader(batch_size=batch_size)
    # ood_loader, _ = fashion_mnist_data_loader(batch_size=batch_size)

    model = PoER(in_channel=in_channel, num_classes=num_classes, dropout=0.1, bit=8, latent_dim=32)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=.5)
    criterion = nn.CrossEntropyLoss()

    best_fpr95 = 0.5
    for epoch in range(epochs):
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        model.train()

        train_feat, train_feat_label = list(), list()  # calculate pair potential energy during inference
        for x, y in tqdm.tqdm(train_loader):
            x = x.float().to(device)
            y = y.long().to(device)
            ranking, cls, recon = model(x)

            # record for inference
            train_feat.append(ranking.detach().cpu())
            train_feat_label.append(y.detach().cpu())

            loss = criterion(cls, y) + ranking_weight * energy_ranking(ranking, y) + recon_weight * recon_loss(recon, x)
            _, predict_cls = torch.max(cls, dim=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            train_acc += get_acc(predict_cls, y) / len(train_loader)

        train_feat = torch.cat(train_feat, dim=0)
        train_feat_label = torch.cat(train_feat_label, dim=0)

        scheduler.step()

        model.eval()
        id_ood_label, id_ood_score = list(), list()
        with torch.no_grad():
            # in distribution data
            for x, y in tqdm.tqdm(val_loader):
                x = x.float().to(device)
                y = y.long().to(device)
                ranking, cls, recon = model(x)
                conf = calculate_cls_conf(cls).to(device)
                dist = calculate_recon_error(recon, x).to(device)
                potential_energy = calculate_pair_potential(train_feat, train_feat_label, ranking, num_classes).to(device)
                score = conf - recon_weight * dist - ranking_weight * potential_energy
                id_ood_label.extend([1] * len(score))
                id_ood_score.extend(score.detach().cpu().numpy().tolist())

                loss = criterion(cls, y) + ranking_weight * energy_ranking(ranking, y) + recon_weight * recon_loss(recon, x)
                _, predict_cls = torch.max(cls, dim=-1)
                val_loss += loss.item() / len(val_loader)
                val_acc += get_acc(predict_cls, y) / len(val_loader)
            
            # out of distribution data
            count = 0
            for x, y in tqdm.tqdm(ood_loader):
                count += 1
                x = x.float().to(device)
                ranking, cls, recon = model(x)
                conf = calculate_cls_conf(cls).to(device)
                dist = calculate_recon_error(recon, x).to(device)
                potential_energy = calculate_pair_potential(train_feat, train_feat_label, ranking, num_classes).to(device)
                score = conf - recon_weight * dist - ranking_weight * potential_energy
                id_ood_label.extend([0] * len(score))
                id_ood_score.extend(score.detach().cpu().numpy().tolist())
                if count >= len(val_loader):
                    break

            auroc, aupr_in, aupr_out, fpr95, thresh95 = get_metric(np.array(id_ood_score), np.array(id_ood_label))

            if fpr95 < best_fpr95:
                best_fpr95 = fpr95
                os.makedirs(save_path, exist_ok=True)
                model_name = '%s/epoch_%d_fpr95_%.4f.pth' % (save_path, epoch, fpr95)
                torch.save(model.state_dict(), model_name)
            
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)

            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)

            writer.add_scalar('val/auroc', auroc, epoch)
            writer.add_scalar('val/aupr_in', aupr_in, epoch)
            writer.add_scalar('val/aupr_out', aupr_out, epoch)
            writer.add_scalar('val/fpr95', fpr95, epoch)
            writer.add_scalar('val/thresh95', thresh95, epoch)

            print('EPOCH : %03d | Train Loss : %.3f | Train Acc : %.3f | Val Loss : %.3f | '
                  'Val Acc : %.3f | AUROC : %.3f | AUPR In : %.3f | AUPR Out : %.3f | '
                  'FPR95 : %.3f | Thresh95 : %.3f' % (epoch, train_loss, train_acc, val_loss,
                   val_acc, auroc, aupr_in, aupr_out, fpr95, thresh95))
        


if __name__ == '__main__':
    in_channel = 3
    num_classes = 10

    # CIFAR-10 Training
    save_path = './saved_models/cifar10'
    logdir = './tensorboard/PoER/cifar10/'

    # MNIST Training 
    # save_path = './saved_models/mnist'
    # logdir = './tensorboard/PoER/mnist/'

    shutil.rmtree(logdir, True)
    writer = SummaryWriter(logdir=logdir)

    lr = 1e-3
    epochs = 1000
    batch_size = 128
    recon_weight = 1
    ranking_weight = 1
    train(epochs, batch_size, lr)
