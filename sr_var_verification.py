import argparse
from PIL import Image
import pickle
import numpy as np
import os
import blobfile as bf
from skimage.metrics import peak_signal_noise_ratio
import torch as th
import torch.nn.functional as F

from sr_var.base_dataset import center_crop_arr
from sr_var.mean_utils import tile_image
from utils import logger


def read_image_numpy(file_path):
    with bf.BlobFile(file_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    np_image = np.array(pil_image)
    return np_image


def read_pkl_numpy(file_path):
    with open(file_path, "rb") as f:
        np_image = pickle.load(f)
    return np_image


def np2th(np_image):
    """
    Transform numpy.array to torch.Tensor and change the order of dimensions.
    We also add a new dimension so that return tensor with shape of (1, C, H, W).
    """
    th_image = th.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
    return th_image


def compute_mse(arr1, arr2):
    """
    Compute MSE
    """
    return np.mean((arr1 - arr2) ** 2)


def compute_psnr(arr1, arr2):
    """
    Compute PSNR.
    """
    return peak_signal_noise_ratio(arr1, arr2, data_range=arr1.max())


def compute_nmse(gt, pred):
    """
    Compute Normalized Mean Squared Error (NMSE)
    """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def generate_ground_truth_from_ddpm_sr(output_dir, data_info, num_samples):
    for file_name in data_info:
        curr_output_dir = os.path.join(output_dir, file_name, "high_res_samples")
        arr = []

        # if we have generated ground truth, skip following computation.
        if os.path.isfile(os.path.join(output_dir, file_name, "high_res_mean.png")) and \
                os.path.isfile(os.path.join(output_dir, file_name, "high_res_var.pkl")):
            continue

        # read high_res_samples
        for j in range(num_samples):
            file_path = os.path.join(curr_output_dir, f"sample_{j}.png")
            np_image = read_image_numpy(file_path)
            arr.append(np_image)

        arr = np.stack(arr)

        if not os.path.isfile(os.path.join(output_dir, file_name, "high_res_mean.png")):
            # Compute the mean of 100 high res images. We save the uint8 data after clip to [0, 255].
            mean_image = np.clip(np.mean(arr, axis=0), 0, 255)
            Image.fromarray(np.uint8(mean_image)).save(os.path.join(output_dir, file_name, "high_res_mean.png"))

        if not os.path.isfile(os.path.join(output_dir, file_name, "high_res_var.pkl")):
            # Compute the variance of 100 high res images.
            # We directly compute the variace from `arr` with range [0, 255].
            # However, high res images are all uint8 data.
            # We save the result as a pickle file, not as an image.
            var_image = np.var(arr, axis=0)
            with open(os.path.join(output_dir, file_name, "high_res_var.pkl"), "wb") as f:
                pickle.dump(var_image, f)


def mean_compare(ddpm_sr_output_dir, sr_model_output_dir, mean_model_step, data_info):
    mse_eval = []
    psnr_eval = []
    nmse_eval = []
    for file_name in data_info:
        ground_truth_image = read_image_numpy(os.path.join(ddpm_sr_output_dir, file_name, "high_res_mean.png"))
        model_output_image = read_image_numpy(
            os.path.join(sr_model_output_dir, file_name, f"high_res_mean_{mean_model_step}.png")
        )
        mse_eval.append(compute_mse(ground_truth_image, model_output_image))
        psnr_eval.append(compute_psnr(ground_truth_image, model_output_image))
        nmse_eval.append(compute_nmse(ground_truth_image, model_output_image))

    mse = np.mean(mse_eval)
    psnr = np.mean(psnr_eval)
    nmse = np.mean(nmse_eval)
    logger.log("Mean Comparison Results")
    logger.log(f"mse:  {mse:.4f}")
    logger.log(f"psnr: {psnr:.4f}")
    logger.log(f"nmse: {nmse:.4f}")


def var_compare(ddpm_sr_output_dir, sr_model_output_dir, var_model_step, data_info):
    var_error_eval = []
    std_error_eval = []
    for file_name in data_info:
        ground_truth_var = read_pkl_numpy(os.path.join(ddpm_sr_output_dir, file_name, "high_res_var.pkl"))
        model_output_var = read_pkl_numpy(
            os.path.join(sr_model_output_dir, file_name, f"high_res_var_{var_model_step}.pkl")
        )
        model_output_var[np.where(model_output_var < 0)] = 0.
        var_error_eval.append(np.mean(np.abs(ground_truth_var - model_output_var)))
        std_error_eval.append(np.mean(np.abs(np.sqrt(ground_truth_var) - np.sqrt(model_output_var))))

    var_error = np.mean(var_error_eval)
    std_error = np.mean(std_error_eval)
    logger.log("Variance Comparison Results")
    logger.log(f"var error: {var_error:.4f}")
    logger.log(f"std error: {std_error:.4f}")


def save_results_to_tensorboard(
        ddpm_sr_output_dir,
        sr_model_output_dir,
        mean_model_step,
        var_model_step,
        data_dir,
        data_info,
        large_size,
):

    for i, file_name in enumerate(data_info):
        original_file_path = os.path.join(data_dir, f"{file_name}.JPEG")
        with bf.BlobFile(original_file_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        arr = center_crop_arr(pil_image, large_size)
        high_res_gt = np2th(arr).clamp(0, 255).to(th.uint8)

        low_res_gt = np2th(
            read_image_numpy(os.path.join(sr_model_output_dir, file_name, "low_res.png"))
        ).clamp(0, 255).to(th.uint8)

        high_res_mean_gt = np2th(
            read_image_numpy(os.path.join(sr_model_output_dir, file_name, f"high_res_mean_{mean_model_step}.png"))
        ).clamp(0, 255).to(th.uint8)

        var_image_gt = read_pkl_numpy(os.path.join(ddpm_sr_output_dir, file_name, "high_res_var.pkl"))
        std_image_gt = np.sqrt(var_image_gt)
        std_image_gt = np2th(std_image_gt / np.max(std_image_gt) * 255.).clamp(0, 255).to(th.uint8)

        var_image_model = read_pkl_numpy(
            os.path.join(sr_model_output_dir, file_name, f"high_res_var_{var_model_step}.pkl")
        )
        var_image_model[np.where(var_image_model < 0)] = 0.
        std_image_model = np.sqrt(var_image_model)
        std_image_model = np2th(std_image_model / np.max(std_image_model) * 255.).clamp(0, 255).to(th.uint8)

        abs_error_map = th.abs(high_res_gt - high_res_mean_gt)
        abs_error_map = (abs_error_map / th.max(abs_error_map) * 255.).to(th.uint8)

        # The image alignment is:
        # groun truth high res from imagenet | low res | high res mean estimation by mean model
        #   abs(high_res - high_res-mean)  | std of ddpm | std estimation by var model
        multi_images = th.cat([high_res_gt, low_res_gt, high_res_mean_gt,
                               abs_error_map, std_image_gt, std_image_model], dim=0)
        multi_images = tile_image(multi_images, ncols=3, nrows=2)
        logger.get_current().write_image('results', multi_images, i + 1)
        if (i + 1) % 100 == 0:
            logger.log(f"have save {i + 1} results")


def main(args):

    logger.configure(args.log_dir, rank=0, is_distributed=False, is_write=True)
    logger.log("")
    logger.log("theorem 1 verification: super-resolution results comparison")
    logger.log("making device configuration...")  # pretend to make device configuration now :)

    with open(args.data_info_dict_path, "rb") as f:
        to_sample_data_info_dict = pickle.load(f)
    to_sample_data_info_list = []
    for class_name, info_dict in to_sample_data_info_dict.items():
        to_sample_data_info_list.extend(info_dict["file_names"])

    # generate ground truth to verification.
    logger.log("generating ground truth, mean and var...")
    generate_ground_truth_from_ddpm_sr(
        args.ddpm_sr_output_dir,
        to_sample_data_info_list,
        args.num_samples,
    )

    logger.log("mean comparing...")
    mean_compare(
        args.ddpm_sr_output_dir,
        args.sr_model_output_dir,
        args.mean_model_step,
        to_sample_data_info_list,
    )

    logger.log("var comparing...")
    var_compare(
        args.ddpm_sr_output_dir,
        args.sr_model_output_dir,
        args.var_model_step,
        to_sample_data_info_list,
    )

    logger.log("saving results...")
    save_results_to_tensorboard(
        args.ddpm_sr_output_dir,
        args.sr_model_output_dir,
        args.mean_model_step,
        args.var_model_step,
        args.data_dir,
        to_sample_data_info_list,
        args.large_size,
    )

    logger.log("complete comparing.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/imagenet/train")
    parser.add_argument("--data_info_dict_path", type=str, default="data/to_sample_data_info.pkl")
    parser.add_argument("--large_size", type=int, default=256)
    parser.add_argument("--ddpm_sr_output_dir", type=str, default="outputs/ddpm_sr/step_250")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--sr_model_output_dir", type=str, default="outputs/sr_var")
    # these two params are to indicate which model to compare.
    parser.add_argument("--mean_model_step", type=int, default=90000)
    parser.add_argument("--var_model_step", type=int, default=50000)
    parser.add_argument("--log_dir", type=str, default="logs/sr_compare")
    args = parser.parse_args()

    main(args)
