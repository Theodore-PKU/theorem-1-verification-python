import argparse
import numpy as np
import torch as th
import torch.distributed as dist

from utils import dist_util, logger
from sr_var.ddpm_sr_utils import DDPMSRDataset, save_images
from sr_var.ddpm.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    # distributed machines configuration
    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.log_dir, rank, is_distributed, is_write=False)

    # load a trianed ddpm sr_mean_train model and set eval mode.
    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # load dataset which includes images to samples.
    logger.log("loading data...")
    dataset = DDPMSRDataset(
        args.data_dir,
        args.to_sample_images_dict_path,
        args.batch_size,
        args.small_size,
        args.large_size,
        args.class_cond,
        args.output_dir,
        args.num_samples_per_image,
    )

    logger.log(f"need to create samples for {len(dataset)} image files")
    logger.log("creating samples...")
    count_have_sample = 0

    for model_kwargs, image_info_dict in dataset:
        # ytxie: image_info_dict contains "file_name", "y", "num_samples" and "num_have_samples".
        file_name = image_info_dict["file_name"]
        num_samples = image_info_dict["num_samples"]
        num_have_samples = image_info_dict["num_have_samples"]

        logger.log(f"begin sampling for {file_name}")

        all_images = []
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        while len(all_images) * args.batch_size < num_samples:
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, 3, args.large_size, args.large_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            # gather samples in different devices
            if is_distributed:
                all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(all_samples, sample)  # gather not supported with NCCL
            else:
                all_samples = [sample]
            for sample in all_samples:
                all_images.append(sample.cpu().numpy())

            logger.log(f"have generated {len(all_images) * args.batch_size} samples"
                       f" for {file_name}")

        # gather all samples
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: num_samples]

        if dist.get_rank() == 0:
            save_images(model_kwargs["low_res"], arr, args.output_dir, file_name, num_have_samples)
            count_have_sample += 1
            logger.log(f"complete sampling for {file_name}")
            logger.log(f"have created samples for {count_have_sample} image files")

    dist.barrier()
    logger.log("sampling complete\n")


def create_argparser():
    defaults = dict(
        log_dir="",
        output_dir="",
        data_dir="",
        model_path="",
        to_sample_images_dict_path="",
        num_samples_per_image=100,
        batch_size=1,
        use_ddim=False,
        clip_denoised=True,
        local_rank=0,  # for distributed device, can be delete?
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # create parser args by defaults
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
