import torchvision.transforms as transforms


def get_transform(size=224):
    # data augmentation
    jitter = 0.4
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, (0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=min(0.5, jitter)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


if __name__ == '__main__':
    train_transform, val_transform = get_transform()