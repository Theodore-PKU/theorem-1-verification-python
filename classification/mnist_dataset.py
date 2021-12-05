"""
Dataset for MNIST data
"""

import torch.utils.data as data
import torchvision.transforms as transforms
import os
import torch
import pickle


class MNISTDataset(data.Dataset):

    def __init__(self, data_dir, noise_label_file=None, train=True):
        if train:
            fname = os.path.join(data_dir, 'training.pt')
        else:
            fname = os.path.join(data_dir, 'test.pt')
        image_set, cat_set = torch.load(fname)
        if noise_label_file:
            with open(os.path.join(data_dir, noise_label_file), "rb") as f:
                self.noisy_cat_set = pickle.load(f)
        else:
            self.noisy_cat_set = None

        self.dataset = (image_set, cat_set)
        self.transform_func = transforms.Compose(
            [PreProcess()]
        )

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, index):
        image, category = self.dataset[0][index], self.dataset[1][index]
        if self.noisy_cat_set is not None:
            noisy_category = torch.tensor(self.noisy_cat_set[index][0])
            prob = torch.tensor(self.noisy_cat_set[index][1])
        else:
            noisy_category = category
            prob = torch.zeros(10)
            prob[int(category)] = 1.
        image = self.transform_func(image)
        return image, category, noisy_category, prob


class PreProcess(object):
    def __call__(self, pic):
        pic = pic / 255.
        return pic.unsqueeze(0)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def load_data(data_dir, batch_size, noise_label_file=None, is_distributed=False, is_train=True, to_train=True):
    dataset = MNISTDataset(data_dir, noise_label_file=noise_label_file, train=is_train)

    data_sampler = None
    if is_distributed:
        data_sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(data_sampler is None) and to_train,
        sampler=data_sampler,
        num_workers=1,
        drop_last=to_train,
        pin_memory=True,
    )

    return loader
