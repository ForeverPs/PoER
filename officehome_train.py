import os
import tqdm
import torch
import shutil
import argparse
from eval import get_acc
from torch import nn, optim
from model import ResNetCls
from loss import energy_ranking
import torch.distributed as dist
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from data_transform import get_transform
from officehome_data import get_dg_dataset


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    if opt.local_rank == 0 and opt.build_tensorboard:
        shutil.rmtree(opt.logdir, True)
        writer = SummaryWriter(logdir=opt.logdir)
        opt.build_tensorboard = False
    
    dist.init_process_group(backend='nccl', init_method=opt.init_method, world_size=opt.n_gpus)

    batch_size = opt.batch_size
    device = torch.device('cuda', opt.local_rank if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))

    train_set, val_set, test_set = get_dg_dataset(train_transform, val_transform, source_domains=opt.source, target_domains=opt.target)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=36)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=24)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=24)
    
    model = ResNetCls(depth=opt.depth, num_classes=opt.num_classes)

    if opt.local_rank == 0:
        try:
            model.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)
        except:
            print('Training from scratch...')

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.poer_set, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(opt.epoch):
        train_loader.sampler.set_epoch(epoch)

        # only tqdm in rank 0
        if opt.local_rank == 0:
            data_loader = tqdm.tqdm(train_loader)
        else:
            data_loader = train_loader
        
        train_loss, val_loss, test_loss = 0, 0, 0
        train_acc, val_acc, test_acc = 0, 0, 0

        model.train()
        for x, y, d in data_loader:
            x, y, d = x.float().to(device), y.long().to(device), d.long().to(device)
            feats, predict = model(x)
            vanilla_loss = criterion(predict, y)
            if opt.poer:
                poer_loss = energy_ranking(feats, y, d)
                poer_weight = 0.1 if epoch < opt.poer_set else 0.2
                loss = vanilla_loss + poer_weight * poer_loss
            else:
                loss = vanilla_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predict_cls = torch.max(predict, dim=-1)
            train_acc += get_acc(predict_cls, y)

        # update learning rate
        scheduler.step()

        if opt.local_rank == 0 and epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                for x, y, d in tqdm.tqdm(val_loader):
                    x, y, d = x.float().to(device), y.long().to(device), d.long().to(device)
                    feats, predict = model(x)
                    vanilla_loss = criterion(predict, y)
                    if opt.poer:
                        poer_loss = energy_ranking(feats, y, d)
                        poer_weight = 0.1 if epoch < opt.poer_set else 0.2
                        loss = vanilla_loss + poer_weight * poer_loss
                    else:
                        loss = vanilla_loss

                    val_loss += loss.item()
                    _, predict_cls = torch.max(predict, dim=-1)
                    val_acc += get_acc(predict_cls, y)
                
                for x, y, d in tqdm.tqdm(test_loader):
                    x, y, d = x.float().to(device), y.long().to(device), d.long().to(device)
                    feats, predict = model(x)
                    loss = criterion(predict, y)
                    test_loss += loss.item()
                    _, predict_cls = torch.max(predict, dim=-1)
                    test_acc += get_acc(predict_cls, y)

            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader)

            val_loss = val_loss / len(val_loader)
            val_acc = val_acc / len(val_loader)

            test_loss = test_loss / len(test_loader)
            test_acc = test_acc / len(test_loader)

            print('EPOCH : %03d | Train Loss : %.4f | Train Acc : %.4f | Val Loss : %.4f | Val Acc : %.4f | '
                  'Test Loss : %.4f | Test Acc : %.4f'
                % (epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

            compare_acc = val_acc - val_loss
            if compare_acc >= opt.best_acc and epoch > opt.min_epoch:
                opt.best_acc = compare_acc
                model_name = 'epoch_%d_val_%.3f_test_%.3f.pth' % (epoch, val_acc, test_acc)
                os.makedirs(opt.save_path, exist_ok=True)
                torch.save(model.module.state_dict(), '%s/%s' % (opt.save_path, model_name))

            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)

            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)

            writer.add_scalar('test/loss', test_loss, epoch)
            writer.add_scalar('test/acc', test_acc, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AAAI PoER')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--init_method', default='env://')
    parser.add_argument('--n_gpus', type=int, default=8)
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--build_tensorboard', type=bool, default=True)

    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=65)
    parser.add_argument('--poer', type=bool, default=True)
    parser.add_argument('--poer_set', type=int, default=70)
    parser.add_argument('--min_epoch', type=int, default=10)
    parser.add_argument('--best_acc', type=float, default=-10)
    parser.add_argument('--source', type=list, default=['Art', 'Clipart', 'Product'])
    parser.add_argument('--target', type=list, default=['Real_World'])
    parser.add_argument('--logdir', type=str, default='./tensorboard/res18_poer/OfficeHome/Real_World/res18_224_run0')
    parser.add_argument('--save_path', type=str, default='./saved_models/res18_poer/OfficeHome/Real_World/res18_224_run0')
    parser.add_argument('--checkpoint', type=str, default=None)

    opt = parser.parse_args()
    if opt.local_rank == 0:
        print('opt:', opt)
    
    # data augmentation
    train_transform, val_transform = get_transform(size=opt.size)
    main(opt)


# using following script to train the model
# python -m torch.distributed.launch --nproc_per_node=8 officehome_train.py --n_gpus=8
