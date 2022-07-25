import os
import tqdm
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

domain2label = {'CALTECH': 0, 'LABELME': 1, 'PASCAL': 2, 'SUN': 3}


def load_train_val_test_pairs(source_domains=None, target_domains=None):
    if source_domains is None:
        source_domains = ['CALTECH', 'LABELME', 'PASCAL']
    if target_domains is None:
        target_domains = ['SUN']

    data_path = './data/VLCS'
    train_pairs, val_pairs, test_pairs = list(), list(), list()
    for domain in source_domains:
        domain_label = domain2label[domain]
        for cat in range(5):
            train_path = '%s/%s/train/%s' % (data_path, domain, cat)
            val_path = '%s/%s/crossval/%s' % (data_path, domain, cat)
            train_img_names = ['%s/%s' % (train_path, img_name) for img_name in os.listdir(train_path)]
            val_img_names = ['%s/%s' % (val_path, img_name) for img_name in os.listdir(val_path)]
        
            for img_name in train_img_names:
                train_pairs.append((img_name, int(cat), int(domain_label)))
            
            for img_name in val_img_names:
                val_pairs.append((img_name, int(cat), int(domain_label)))
    
    for domain in target_domains:
        domain_label = domain2label[domain]
        for cat in range(5):
            test_path = '%s/%s/test/%s' % (data_path, domain, cat)
            test_img_names = ['%s/%s' % (test_path, img_name) for img_name in os.listdir(test_path)]
            
            for img_name in test_img_names:
                test_pairs.append((img_name, int(cat), int(domain_label)))
    
    
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
    train_transform, val_transform = get_transform(resize=227, size=224)
    train_set, val_set, test_set = get_dg_dataset(train_transform, val_transform)
    train_loader = DataLoader(train_set, batch_size=24, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_set, batch_size=24, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=24, shuffle=False, num_workers=12)

    print(len(train_set), len(val_set), len(test_set))

    for x, y, d in tqdm.tqdm(train_loader):
        print(x.shape, y.shape, d.shape)
    
    for x, y, d in tqdm.tqdm(val_loader):
        print(x.shape, y.shape, d.shape)
    
    for x, y, d in tqdm.tqdm(test_loader):
        print(x.shape, y.shape, d.shape)

    # train_pairs, val_pairs, test_pairs = load_train_val_test_pairs()
    # print(train_pairs[0], val_pairs[0], test_pairs[0])