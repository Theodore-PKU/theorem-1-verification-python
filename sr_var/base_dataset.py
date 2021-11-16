"""
Basic dataset for ImageNet data
"""

from PIL import Image
import blobfile as bf
import random
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def load_data(
    *,
    data_dir,
    data_info_dict_path,
    batch_size,
    image_size,
    random_flip=True,
    is_distributed=False,
    is_train=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory saving high resolution images.
    :param data_info_dict_path: a pickle file path which contains a dict.
    :param image_size: image size of high resolution.
    :param batch_size: the batch size of each returned pair.
    :param random_flip: whether to flip image when training.
    :param is_distributed: if True, use DistributedSampler.
    :param is_train: if True, shuffle the dataset when training.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    with open(data_info_dict_path, "rb") as f:
        data_info_dict = pickle.load(f)

    dataset = ImageDataset(
        data_dir,
        data_info_dict,
        image_size,
        random_flip=random_flip,
    )

    data_sampler = None
    if is_distributed:
        data_sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(data_sampler is None) and is_train,
        sampler=data_sampler,
        num_workers=1,
        drop_last=is_train,
        pin_memory=True,
    )

    while True:
        yield from loader


# ytxie: This is the key part in this code.
# ytxie: Since it is to divide the dataset by hand and we are not familiar with this way, we modify
# ytxie: this class.
class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        data_info_dict,
        resolution,
        random_flip=True,
    ):
        super().__init__()
        assert data_dir
        self.data_dir = data_dir
        self.resolution = resolution
        self.random_flip = random_flip
        self.data_info_list = []
        for class_name, info_dict in data_info_dict.items():
            self.data_info_list.extend(info_dict["file_names"])

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):
        file_name = self.data_info_list[idx]
        file_path = os.path.join(self.data_dir, f"{file_name}.JPEG")
        with bf.BlobFile(file_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        arr = center_crop_arr(pil_image, self.resolution)
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = arr.astype(np.float32) / 127.5 - 1
        
        return np.transpose(arr, [2, 0, 1]), file_name


# ytxie: This function is to crop iamges at the center location and is not used directly.
# ytxie: The parameter `image_size` is of int type.
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    # return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]  # source
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]  # new
