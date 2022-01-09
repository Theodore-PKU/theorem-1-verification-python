from sr_var_verification import *


def main(args):
    logger.configure(args.log_dir, rank=0, is_distributed=False, is_write=False)

    with open(args.data_info_dict_path, "rb") as f:
        to_sample_data_info_dict = pickle.load(f)
    to_sample_data_info_list = []
    for class_name, info_dict in to_sample_data_info_dict.items():
        to_sample_data_info_list.extend(info_dict["file_names"])

    mse_dict = {
        "ddpm_sr_mean_vs_groud_truth": [],
        "model_mean_vs_groud_truth": [],
        "model_mean_vs_ddpm_sr_mean": [],
        "model_var_vs_ddpm_sr_var": [],
    }
    psnr_dict = {
        "ddpm_sr_mean_vs_groud_truth": [],
        "model_mean_vs_groud_truth": [],
        "model_mean_vs_ddpm_sr_mean": [],
        "model_var_vs_ddpm_sr_var": [],
    }
    nmse_dict = {
        "ddpm_sr_mean_vs_groud_truth": [],
        "model_mean_vs_groud_truth": [],
        "model_mean_vs_ddpm_sr_mean": [],
        "model_var_vs_ddpm_sr_var": [],
    }
    for file_name in to_sample_data_info_list:
        original_file_path = os.path.join(args.data_dir, f"{file_name}.JPEG")
        with bf.BlobFile(original_file_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        arr = center_crop_arr(pil_image, args.large_size)
        ground_truth_image = th.from_numpy(arr).clamp(0, 255).to(th.uint8).numpy()

        ddpm_sr_mean_image = read_image_numpy(os.path.join(args.ddpm_sr_output_dir, file_name, "high_res_mean.png"))
        ddpm_sr_var = read_pkl_numpy(os.path.join(args.ddpm_sr_output_dir, file_name, "high_res_var.pkl"))

        model_mean_image = read_image_numpy(
            os.path.join(args.sr_model_output_dir, file_name, f"high_res_mean_{args.mean_model_step}.png")
        )
        model_var = read_pkl_numpy(
            os.path.join(args.sr_model_output_dir, file_name, f"high_res_var_{args.var_model_step}.pkl")
        )
        model_var[np.where(model_var < 0)] = 0.

        ddpm_sr_std = np.sqrt(ddpm_sr_var)
        model_std = np.sqrt(model_var)

        for matrix in [
            "ddpm_sr_mean_vs_groud_truth",
            "model_mean_vs_groud_truth",
            "model_mean_vs_ddpm_sr_mean",
            "model_var_vs_ddpm_sr_var",
        ]:
            if matrix == "ddpm_sr_mean_vs_groud_truth":
                a = ddpm_sr_mean_image
                b = ground_truth_image
            elif matrix == "model_mean_vs_groud_truth":
                a = model_mean_image
                b = ground_truth_image
            elif matrix == "model_mean_vs_ddpm_sr_mean":
                a = model_mean_image
                b = ddpm_sr_mean_image
            else:
                a = model_std
                b = ddpm_sr_std
            mse_dict[matrix].append(compute_mse(a, b))
            psnr_dict[matrix].append(compute_psnr(a, b))
            nmse_dict[matrix].append(compute_nmse(a, b))
    for matrix in [
        "ddpm_sr_mean_vs_groud_truth",
        "model_mean_vs_groud_truth",
        "model_mean_vs_ddpm_sr_mean",
        "model_var_vs_ddpm_sr_var",
    ]:
        logger.log(f"{matrix}: MSE,  {np.mean(mse_dict[matrix]):.4f}")
        logger.log(f"{matrix}: PSNR, {np.mean(psnr_dict[matrix]):.4f}")
        logger.log(f"{matrix}: NMSE, {np.mean(nmse_dict[matrix]):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/imagenet/train")
    parser.add_argument("--data_info_dict_path", type=str, default="data/sr_imagenet/to_sample_data_info.pkl")
    parser.add_argument("--large_size", type=int, default=256)
    parser.add_argument("--ddpm_sr_output_dir", type=str, default="outputs/ddpm_sr/64_256_step_250")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--sr_model_output_dir", type=str, default="outputs/sr_var/64_256")
    # these two params are to indicate which model to compare.
    parser.add_argument("--mean_model_step", type=int, default=90000)
    parser.add_argument("--var_model_step", type=int, default=50000)
    parser.add_argument("--log_dir", type=str, default="logs/sr_compare/64_256")
    args = parser.parse_args()

    main(args)
