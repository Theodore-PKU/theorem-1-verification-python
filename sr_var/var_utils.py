"""
Training, Testing and dataset for super-resolution var estimation task
"""

import os
import pickle
import torch as th
import torch.nn.functional as F

from utils.train_util import TrainLoop
from utils.test_util import TestLoop
from utils import dist_util, logger
from .mean_utils import transform_to_uint8, tile_image
from .base_dataset import load_data


# ================================================================
# TrainLoop and TestLoop
# ================================================================

class SRVarTrainLoop(TrainLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _post_process(self, micro_input, micro_label, micro_output, i):
        if self.step % (self.log_interval * 100) == 0 and i == 0:
            micro_low_res = transform_to_uint8(micro_input[:, 0:3, ...])
            micro_high_res_mean = transform_to_uint8(micro_input[:, 3:6, ...])
            micro_label = var_transform_std_image(micro_label)
            micro_output = var_transform_std_image(micro_output)
            ncols = len(micro_input)
            low_res_tiled_images = tile_image(micro_low_res, ncols=ncols, nrows=1)
            high_res_mean_tiled_images = tile_image(micro_high_res_mean, ncols=ncols, nrows=1)
            label_tiled_images = tile_image(micro_label, ncols=ncols, nrows=1)
            output_tiled_images = tile_image(micro_output, ncols=ncols, nrows=1)
            all_tiled_images = th.cat(
                [low_res_tiled_images, high_res_mean_tiled_images, output_tiled_images, label_tiled_images], dim=1
            )
            logger.get_current().write_image(
                'train_images',
                all_tiled_images,
                self.step
            )


class SRVarTestLoop(TestLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _post_process(self, batch, output, batch_kwargs):
        # save output as pkl file to output_dir.
        # `output` var is computed in range of [-1, 1], while when we generate ground truth, var is
        # computed in range [0, 255], so we need to multiply by 127.5 ** 2.
        # some value may be lower than 0.
        output_pkl = output.permute(0, 2, 3, 1) * (127.5 ** 2)  # (B, H, W, C)
        for i in range(len(output_pkl)):
            file_name = batch_kwargs[i]
            var_output = output_pkl[i].detach().cpu().numpy()

            output_dir = os.path.join(self.output_dir, file_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"high_res_var_{self.resume_step}.pkl"), "wb") as f:
                pickle.dump(var_output, f)

        # save to tensorboard
        output = var_transform_std_image(output)
        batch = transform_to_uint8(batch[:, 0:3, ...])
        ncols = len(batch)
        input_tiled_images = tile_image(batch, ncols=ncols, nrows=1)
        output_tiled_images = tile_image(output, ncols=ncols, nrows=1)
        all_tiled_images = th.cat(
            [input_tiled_images, output_tiled_images], dim=1
        )
        logger.get_current().write_image(
            'test_images',
            all_tiled_images,
            self.step,
        )


def var_transform_std_image(var):
    """
    Transform var tensor to std image which value range is [0, 255]
    :param var: torch.Tensor, (B, C, H, W)
    :return: torch.Tensor, transformed std
    """
    var[th.where(var < 0)] = 0.
    var = th.sqrt(var)
    # The following computation guarantees that the value range is correct.
    # We change `var` to [-1, 1] so that it is consistent with `transform_to_uint8`.
    for i in range(len(var)):
        var[i] = var[i] / th.max(var[i])
    var = var * 2. - 1.
    var = transform_to_uint8(var)
    return var


# ================================================================
# Dataset
# ================================================================

def load_train_data(
        data_dir,
        data_info_dict_path,
        batch_size,
        large_size,
        small_size,
        mean_model,
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
    mean_model.eval()
    for large_batch, _ in data:
        small_batch = F.interpolate(large_batch, small_size, mode="area")
        small_batch = F.interpolate(small_batch, (large_size, large_size), mode="bilinear").to(dist_util.dev())
        with th.no_grad():
            mean_output = mean_model(small_batch)
        label = (large_batch.to(dist_util.dev()) - mean_output) ** 2
        batch = th.cat([small_batch, mean_output], dim=1)
        yield batch, label


def load_test_data(
        data_dir,
        data_info_dict_path,
        batch_size,
        large_size,
        small_size,
        mean_model,
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
    mean_model.eval()
    file_name_list = []
    stop_flag = False
    for large_batch, file_name_batch in data:
        small_batch = F.interpolate(large_batch, small_size, mode="area")
        small_batch = F.interpolate(small_batch, (large_size, large_size), mode="bilinear").to(dist_util.dev())
        with th.no_grad():
            mean_output = mean_model(small_batch)
        batch = th.cat([small_batch, mean_output], dim=1)
        yield batch, file_name_batch
        for file_name in file_name_batch:
            if file_name in file_name_list:
                stop_flag = True
                break
        if stop_flag:
            break
        file_name_list.extend(file_name_batch)
