import os
import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

name2label = {name: i for i, name in enumerate(os.listdir('./data/NICO/Animal_Vehicle/'))}

domain2label, count = dict(), 0
for cat in list(name2label.keys()):
    cat_path = './data/NICO/Animal_Vehicle/%s' % cat
    domains = os.listdir(cat_path)
    for domain in domains:
        if domain not in list(domain2label.keys()):
            domain2label[domain] = count
            count += 1


def random_split(val_ratio=0.1, test_domains=2):
    data_root = './data/NICO/Animal_Vehicle'
    datalist = './data/NICO/datalist'
    os.makedirs(datalist, exist_ok=True)

    train_pairs, val_pairs, test_pairs = list(), list(), list()
    for cat, cat_label in name2label.items():
        cat_path = '%s/%s' % (data_root, cat)
        domains = list(os.listdir(cat_path))
        target_domains = np.random.choice(domains, size=test_domains, replace=False)
        source_domains = list(set(domains).difference(set(target_domains)))
        for domain in source_domains:
            domain_path = '%s/%s' % (cat_path, domain)
            img_names = ['%s/%s' % (domain_path, img_name) for img_name in os.listdir(domain_path)]
            val_num = int(len(img_names) * val_ratio) + 1
            val_names = np.random.choice(img_names, size=val_num, replace=False)
            train_names = list(set(img_names).difference(set(val_names)))
            cat_train_pairs = [(img_name, str(cat_label), str(domain2label[domain])) for img_name in train_names]
            cat_val_pairs = [(img_name, str(cat_label), str(domain2label[domain])) for img_name in val_names]
            train_pairs.extend(cat_train_pairs)
            val_pairs.extend(cat_val_pairs)

        for domain in target_domains:
            domain_path = '%s/%s' % (cat_path, domain)
            img_names = ['%s/%s' % (domain_path, img_name) for img_name in os.listdir(domain_path)]
            cat_test_pairs = [(img_name, str(cat_label), str(domain2label[domain])) for img_name in img_names]
            test_pairs.extend(cat_test_pairs)

    train_path = '%s/train.txt' % datalist
    val_path = '%s/val.txt' % datalist
    test_path = '%s/test.txt' % datalist
    with open(train_path, 'w') as f:
        for i, pair in enumerate(train_pairs):
            write_line = ','.join(pair) + '\n' if i < len(train_pairs) - 1 else ','.join(pair)
            f.write(write_line)
        f.close()
    
    with open(val_path, 'w') as f:
        for i, pair in enumerate(val_pairs):
            write_line = ','.join(pair) + '\n' if i < len(val_pairs) - 1 else ','.join(pair)
            f.write(write_line)
        f.close()
    
    with open(test_path, 'w') as f:
        for i, pair in enumerate(test_pairs):
            write_line = ','.join(pair) + '\n' if i < len(test_pairs) - 1 else ','.join(pair)
            f.write(write_line)
        f.close()


def load_train_val_test_pairs():
    train_pairs, val_pairs, test_pairs = list(), list(), list()
    train_txt = './data/NICO/datalist/train.txt'
    val_txt = './data/NICO/datalist/val.txt'
    test_txt = './data/NICO/datalist/test.txt'

    with open(train_txt, 'r') as f:
        train_lines = f.readlines()
    with open(val_txt, 'r') as f:
        val_lines = f.readlines()
    with open(test_txt, 'r') as f:
        test_lines = f.readlines()
    
    for line in train_lines:
        try:
            img_name, label, domain_label = line.strip().split(',')
            train_pairs.append((img_name, int(label), int(domain_label)))
        except:
            pass
    
    for line in val_lines:
        try:
            img_name, label, domain_label = line.strip().split(',')
            val_pairs.append((img_name, int(label), int(domain_label)))
        except:
            pass
        
    for line in test_lines:
        try:
            img_name, label, domain_label = line.strip().split(',')
            test_pairs.append((img_name, int(label), int(domain_label)))
        except:
            pass
    
    return train_pairs, val_pairs, test_pairs


class PACSDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        while True:
            try:
                img_name, cat_label, domain_label = self.pairs[index]
                img = Image.open(img_name).convert('RGB')
                break
            except:
                index = (index + 1) % len(self.pairs)
        return self.transform(img), int(cat_label), int(domain_label)


def get_dg_dataset(train_transform, val_transform):
    train_pairs, val_pairs, test_pairs = load_train_val_test_pairs()
    train_set = PACSDataset(train_pairs, train_transform)
    val_set = PACSDataset(val_pairs, val_transform)
    test_set = PACSDataset(test_pairs, val_transform)
    return train_set, val_set, test_set


if __name__ == '__main__':
    # from data_transform import get_transform
    # train_transform, val_transform = get_transform()
    # train_set, val_set, test_set = get_dg_dataset(train_transform, val_transform)
    # train_loader = DataLoader(train_set, batch_size=24, shuffle=True, num_workers=12)
    # val_loader = DataLoader(val_set, batch_size=24, shuffle=False, num_workers=12)
    # test_loader = DataLoader(test_set, batch_size=24, shuffle=False, num_workers=12)

    # print(len(train_set), len(val_set), len(test_set))

    # for x, y, d in tqdm.tqdm(train_loader):
    #     print(x.shape, y.shape, d.shape)
    
    # for x, y, d in tqdm.tqdm(val_loader):
    #     print(x.shape, y.shape, d.shape)
    
    # for x, y, d in tqdm.tqdm(test_loader):
    #     print(x.shape, y.shape, d.shape)

    print(name2label)
    print(domain2label)
    print(len(name2label), len(domain2label))
    # random_split()