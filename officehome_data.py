import os
import tqdm
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

domain2label = {'Art': 0, 'Clipart': 1, 'Product': 2, 'Real_World': 3}
name2label = {name: i for i, name in enumerate(os.listdir('./data/OfficeHome/Art/'))}


def random_split(val_ratio):
    data_root = './data/OfficeHome'
    datalist = './data/OfficeHome/datalist'
    os.makedirs(datalist, exist_ok=True)
    for domain, domain_label in domain2label.items():
        domain_path = '%s/%s' % (data_root, domain)
        domain_train, domain_val = list(), list()
        for cat, cat_label in name2label.items():
            cat_path = '%s/%s' % (domain_path, cat)
            all_names = ['%s/%s' % (cat_path, img_name) for img_name in os.listdir(cat_path)]
            val_num = int(val_ratio * len(all_names)) + 1
            val_names = np.random.choice(all_names, size=val_num, replace=False)
            train_names = list(set(all_names).difference(set(val_names)))
            val_pairs = [(val_name, str(cat_label), str(domain_label)) for val_name in val_names]
            train_pairs = [(train_name, str(cat_label), str(domain_label)) for train_name in train_names]
            domain_train.extend(train_pairs)
            domain_val.extend(val_pairs)
        domain_test = domain_train + domain_val
        train_path = '%s/%s_train.txt' % (datalist, domain)
        val_path = '%s/%s_val.txt' % (datalist, domain)
        test_path = '%s/%s_test.txt' % (datalist, domain)
        with open(train_path, 'w') as f:
            for i, pair in enumerate(domain_train):
                write_line = ' '.join(pair) + '\n' if i < len(domain_train) - 1 else ' '.join(pair)
                f.write(write_line)
            f.close()
        
        with open(val_path, 'w') as f:
            for i, pair in enumerate(domain_val):
                write_line = ' '.join(pair) + '\n' if i < len(domain_val) - 1 else ' '.join(pair)
                f.write(write_line)
            f.close()
        
        with open(test_path, 'w') as f:
            for i, pair in enumerate(domain_test):
                write_line = ' '.join(pair) + '\n' if i < len(domain_test) - 1 else ' '.join(pair)
                f.write(write_line)
            f.close()


def load_train_val_test_pairs(txt_path=None, data_path=None, source_domains=None, target_domains=None):
    if txt_path is None:
        txt_path = './data/OfficeHome/datalist/'
    if data_path is None:
        data_path = './data/'
    if source_domains is None:
        source_domains = ['Art', 'Clipart', 'Product']
    if target_domains is None:
        target_domains = ['Real_World']
    
    train_pairs, val_pairs, test_pairs = list(), list(), list()
    for domain in source_domains:
        domain_label = domain2label[domain]
        train_txt = txt_path + '%s_train.txt' % domain
        val_txt = txt_path + '%s_val.txt' % domain
        with open(train_txt, 'r') as f:
            train_lines = f.readlines()
        with open(val_txt, 'r') as f:
            val_lines = f.readlines()
        
        for line in train_lines:
            img_name, label, domain_label = line.strip().split(' ')
            train_pairs.append((img_name, int(label), int(domain_label)))
        
        for line in val_lines:
            img_name, label, domain_label = line.strip().split(' ')
            val_pairs.append((img_name, int(label), int(domain_label)))
    
    for domain in target_domains:
        test_txt = txt_path + '%s_test.txt' % domain
        with open(test_txt, 'r') as f:
            test_lines = f.readlines()
        
        for line in test_lines:
            img_name, label, domain_label = line.strip().split(' ')
            test_pairs.append((img_name, int(label), int(domain_label)))
    
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

    for x, y, d in tqdm.tqdm(train_loader):
        print(x.shape, y.shape, d.shape)
    
    for x, y, d in tqdm.tqdm(val_loader):
        print(x.shape, y.shape, d.shape)
    
    for x, y, d in tqdm.tqdm(test_loader):
        print(x.shape, y.shape, d.shape)
