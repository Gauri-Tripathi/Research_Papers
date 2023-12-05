# data.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils import CLASS_NAMES


def get_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 expects 299x299 input
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))  # ImageNet normalization
    ])

    full_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)

    if 'selected_classes' in config and config['selected_classes']:
        selected_idxs = []
        class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        selected_labels = [class_to_idx[c] for c in config['selected_classes']]
        
        for idx, (_, label) in enumerate(full_dataset):
            if label in selected_labels:
                selected_idxs.append(idx)

        full_dataset = Subset(full_dataset, selected_idxs)

    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    return train_loader, val_loader