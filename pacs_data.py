import os
import tqdm
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

domain2label = {'art_painting': 0, 'cartoon': 1, 'photo': 2, 'sketch': 3}
class2label = {'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}


def load_train_val_test_pairs(txt_path=None, data_path=None, source_domains=None, target_domains=None):
    if txt_path is None:
        txt_path = './data/PACS/datalist/PACS/'
    if data_path is None:
        data_path = './data/'
    if source_domains is None:
        source_domains = ['art_painting', 'cartoon', 'photo']
    if target_domains is None:
        target_domains = ['sketch']
    
    train_pairs, val_pairs, test_pairs = list(), list(), list()
    for domain in source_domains:
        domain_label = domain2label[domain]
        train_txt = txt_path + '%s_train_kfold.txt' % domain
        val_txt = txt_path + '%s_crossval_kfold.txt' % domain
        with open(train_txt, 'r') as f:
            train_lines = f.readlines()
        with open(val_txt, 'r') as f:
            val_lines = f.readlines()
        
        for line in train_lines:
            img_name, label = line.strip().split(' ')
            abs_img_name = data_path + img_name
            train_pairs.append((abs_img_name, int(label), domain_label))
        
        for line in val_lines:
            img_name, label = line.strip().split(' ')
            abs_img_name = data_path + img_name
            val_pairs.append((abs_img_name, int(label), domain_label))
    
    for domain in target_domains:
        domain_label = domain2label[domain]
        test_txt = txt_path + '%s_test.txt' % domain
        with open(test_txt, 'r') as f:
            test_lines = f.readlines()
        
        for line in test_lines:
            img_name, label = line.strip().split(' ')
            abs_img_name = data_path + img_name
            test_pairs.append((abs_img_name, int(label), domain_label))
    
    return train_pairs, val_pairs, test_pairs


class PACSDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_name, cat_label, domain_label = self.pairs[index]
        img = Image.open(img_name).convert('RGB')
        return self.transform(img), int(cat_label), int(domain_label)


def get_dg_dataset(train_transform, val_transform, source_domains=None, target_domains=None):
    train_pairs, val_pairs, test_pairs = load_train_val_test_pairs(source_domains=source_domains, target_domains=target_domains)
    train_set = PACSDataset(train_pairs, train_transform)
    val_set = PACSDataset(val_pairs, val_transform)
    test_set = PACSDataset(test_pairs, val_transform)
    return train_set, val_set, test_set


if __name__ == '__main__':
    from data_transform import get_transform
    train_transform, val_transform = get_transform()
    train_set, val_set, test_set = get_dg_dataset(train_transform, val_transform)
    train_loader = DataLoader(train_set, batch_size=24, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_set, batch_size=24, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=24, shuffle=False, num_workers=12)

    print(len(train_set), len(val_set), len(test_set))

    # for x, y, d in tqdm.tqdm(train_loader):
    #     print(x.shape, y.shape, d.shape)
    
    # for x, y, d in tqdm.tqdm(val_loader):
    #     print(x.shape, y.shape, d.shape)
    
    # for x, y, d in tqdm.tqdm(test_loader):
    #     print(x.shape, y.shape, d.shape)


