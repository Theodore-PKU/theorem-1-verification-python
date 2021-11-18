"""
Training, Testing and dataset for super-resolution mean estimation task
"""

import os
from PIL import Image
import torch as th
import torch.nn.functional as F

from utils.train_util import TrainLoop
from utils.test_util import TestLoop
from utils import logger
from .base_dataset import load_data


# ================================================================
# TrainLoop and TestLoop
# ================================================================

class SRMeanTrainLoop(TrainLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _post_process(self, micro_input, micro_label, micro_output, i):
        if self.step % (self.log_interval * 100) == 0 and i == 0:
            micro_input = transform_to_uint8(micro_input)
            micro_label = transform_to_uint8(micro_label)
            micro_output = transform_to_uint8(micro_output)
            ncols = len(micro_input)
            input_tiled_images = tile_image(micro_input, ncols=ncols, nrows=1)
            label_tiled_images = tile_image(micro_label, ncols=ncols, nrows=1)
            output_tiled_images = tile_image(micro_output, ncols=ncols, nrows=1)
            # image alignment:
            # low res | high res mean estimation | high res label
            all_tiled_images = th.cat(
                [input_tiled_images, output_tiled_images, label_tiled_images], dim=1
            )
            logger.get_current().write_image(
                'train_images',
                all_tiled_images,
                self.step
            )


class SRMeanTestLoop(TestLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _post_process(self, batch, output, batch_kwargs):
        """
        Save results.
        :param batch: model input.
        :param output: model output.
        :param batch_kwargs: a list of file_names.
        :return:
        """
        # save to tensorboard
        batch = transform_to_uint8(batch)
        output = transform_to_uint8(output)
        ncols = len(batch)
        input_tiled_images = tile_image(batch, ncols=ncols, nrows=1)
        output_tiled_images = tile_image(output, ncols=ncols, nrows=1)
        # image alignment:
        # low res | high res mean estimation
        all_tiled_images = th.cat(
            [input_tiled_images, output_tiled_images], dim=1
        )
        logger.get_current().write_image(
            'test_images',
            all_tiled_images,
            self.step,
        )

        # save to output_dir
        input_images = []
        batch = batch.permute(0, 2, 3, 1)
        batch = batch.contiguous()
        for img in batch:
            input_images.append(img.detach().cpu().numpy())

        output_images = []
        output = output.permute(0, 2, 3, 1)
        output = output.contiguous()
        for img in output:
            output_images.append(img.detach().cpu().numpy())

        for i in range(len(input_images)):
            low_res = Image.fromarray(input_images[i])
            mean_img = Image.fromarray(output_images[i])
            file_name = batch_kwargs[i]

            output_dir = os.path.join(self.output_dir, file_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            low_res.save(os.path.join(output_dir, f"low_res.png"))
            mean_img.save(os.path.join(output_dir, f"high_res_mean_{self.resume_step}.png"))


def transform_to_uint8(image):
    return ((image + 1) * 127.5).clamp(0, 255).to(th.uint8)


def tile_image(batch_image, ncols, nrows):
    assert ncols * nrows == batch_image.shape[0]
    _, channels, height, width = batch_image.shape
    batch_image = batch_image.view(nrows, ncols, channels, height, width)
    batch_image = batch_image.permute(2, 0, 3, 1, 4)
    batch_image = batch_image.contiguous().view(channels, nrows * height, ncols * width)
    return batch_image


# ================================================================
# Dataset
# ================================================================

def load_train_data(
        data_dir,
        data_info_dict_path,
        batch_size,
        large_size,
        small_size,
        is_distributed=False,
):
    data = load_data(
        data_dir=data_dir,
        data_info_dict_path=data_info_dict_path,
        batch_size=batch_size,
        image_size=large_size,
        random_flip=True,
        is_distributed=is_distributed,
        is_train=True,
    )
    for large_batch, _ in data:
        small_batch = F.interpolate(large_batch, small_size, mode="area")
        small_batch = F.interpolate(small_batch, (large_size, large_size), mode="bilinear")
        yield small_batch, large_batch


def load_test_data(
        data_dir,
        data_info_dict_path,
        batch_size,
        large_size,
        small_size,
        is_distributed=False,
):
    data = load_data(
        data_dir=data_dir,
        data_info_dict_path=data_info_dict_path,
        batch_size=batch_size,
        image_size=large_size,
        random_flip=False,
        is_distributed=is_distributed,
        is_train=False,
    )
    file_name_list = []
    stop_flag = False
    for large_batch, file_name_batch in data:
        small_batch = F.interpolate(large_batch, small_size, mode="area")
        small_batch = F.interpolate(small_batch, (large_size, large_size), mode="bilinear")
        yield small_batch, file_name_batch
        for file_name in file_name_batch:
            if file_name in file_name_list:
                stop_flag = True
                break
        if stop_flag:
            break
        file_name_list.extend(file_name_batch)
