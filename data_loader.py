import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


class CelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[2:]
        # random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)

            if (i+1) >= 182638:
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
            label = self.train_labels[index]
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
            label = self.test_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data


class TestDataset(Dataset):
    def __init__(self, image_path, transform):
        self.image_path = image_path
        self.transform = transform
        self.list = self.get_list(image_path)
        self.num_data = len(self.list)

    def get_list(self, image_path):
        return os.listdir(image_path)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.list[index]))
        return self.transform(image)

    def __len__(self):
        return self.num_data


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class ImageMaskFolder(Dataset):
    def __init__(self, root, transform=None, mask_transform=None, target_transform=None, mode='train'):
        classes, class_to_idx = find_classes(os.path.join(root, mode, 'Image'))
        imgs = make_dataset(os.path.join(root, mode, 'Image'), class_to_idx)
        if mode == 'train':
            self.masks = make_dataset(os.path.join(root, mode, 'Mask'), class_to_idx)
        self.mode = mode
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.mask_transform = mask_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.mode == 'train':
            path, _ = self.masks[index]
            mask = Image.open(path).convert('RGB')
            mask = self.mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.mode == 'train':
            return img, mask, target
        else:
            return img, target


    def __len__(self):
        return len(self.imgs)


def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train', with_mask=False):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_celebA = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if with_mask:
            mask_transform = transforms.Compose([
                # transforms.CenterCrop(crop_size),
                transforms.Scale(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
    elif mode == 'test' or mode == 'evaluate' or mode == 'vis':
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_celebA = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif mode == 'demo':
        transform = transforms.Compose([
            transforms.Scale((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if mode == 'demo':
        dataset = TestDataset(image_path, transform)
    elif dataset == 'CelebA':
        dataset = CelebDataset(image_path, metadata_path, transform_celebA, mode)
    else:
        if with_mask:
            dataset = ImageMaskFolder(image_path, transform, mask_transform, None, mode)
        else:
            dataset = ImageFolder(os.path.join(image_path, mode), transform)
    print(len(dataset))
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
