import torch

from torch.utils.data import Subset
from torch.utils.data import Dataset
import torchvision.datasets as datasets

import torchvision.transforms as T
import torchvision.transforms as transforms

import os
import numpy as np

def get_dataloader(batch_size, num_workers, image_size, num_folds):
    return FinalProjectDataset(batch_size, num_workers, image_size, num_folds)


def kfold_split(dataset, num_folds=10, img_per_cls_per_fold=1, seed=42):
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    rng = np.random.default_rng(seed)
    needed = num_folds * img_per_cls_per_fold

    val_per_fold = [[] for _ in range(num_folds)]

    for label in sorted(class_indices.keys()):
        indices = rng.permutation(class_indices[label]).tolist()
        n = len(indices)
        n_val = min(n, needed)
        

        chosen_slots = rng.permutation(needed)[:n_val].tolist()
        for slot, img_idx in zip(chosen_slots, indices[:n_val]):
            fold = slot // img_per_cls_per_fold
            val_per_fold[fold].append(img_idx)

    all_indices = set(range(len(dataset.samples)))
    folds = []
    for fold in range(num_folds):
        val = val_per_fold[fold]
        train = sorted(all_indices - set(val))
        folds.append((train, val))

    return folds


class SortedImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        lst = [i for i in os.listdir(directory) if i != '.DS_Store']
        classes = sorted(lst, key=lambda x: int(x))
        class_to_idx = {cls: int(cls) for cls in classes}
        return classes, class_to_idx


class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class FinalProjectDataset():
    def __init__(self, batch_size, num_workers, image_size, num_folds):
        mean = [0.5, 0.5, 0.5]
        std  = [0.5, 0.5, 0.5]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_folds = num_folds
        self.img_per_cls_per_fold = 1

        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.train_transform = T.Compose([
            T.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.6, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        ])

        full_data = SortedImageFolder(root='./data/train', transform=None)

        # expose for use in training / routing network
        self.full_data = full_data
        self.class_to_idx = full_data.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Pre-compute all folds once 
        self.folds = kfold_split(
            full_data,
            num_folds=self.num_folds,
            img_per_cls_per_fold=self.img_per_cls_per_fold,
        )

        self.fold = None
        self._build_loaders(0)

    def _build_loaders(self, fold):
        assert 0 <= fold < self.num_folds, \
            f"fold must be in [0, {self.num_folds}), got {fold}"
        self.fold = fold
        train_idx, val_idx = self.folds[fold]

        self.train_subset = Subset(self.full_data, train_idx)
        self.val_subset   = Subset(self.full_data, val_idx)

        self.trainset = TransformDataset(self.train_subset, transform=self.train_transform)
        self.valset   = TransformDataset(self.val_subset,   transform=self.val_transform)

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers)

    def set_fold(self, fold):
        print(f'** switching to fold {fold} **')
        self._build_loaders(fold)