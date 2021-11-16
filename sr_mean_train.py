"""
Training process for super-resolution mean estimation
"""

import argparse
import os

from utils import dist_util, logger
from utils.script_util import add_dict_to_argparser, args_to_dict, load_args_dict, save_args_dict
from sr_var.mean_utils import SRMeanTrainLoop, load_train_data
from sr_var.script_util import mean_model_defaults, create_mean_model


INITIAL_LOG_LOSS_SCALE = 20.0


def main():
    args = create_argparser().parse_args()

    # distributed machines configuration
    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.log_dir, rank, is_distributed, is_write=True)
    logger.log("")
    logger.log("theorem 1 verification: super-resolution mean estimation training")
    logger.log("making device configuration...")  # pretend to make device configuration now :)

    logger.log("creating model...")
    if args.resume_checkpoint:
        # load model args which has been created last time
        model_args = load_args_dict(os.path.join(args.log_dir, "mean_model_args.pkl"))
        model = create_mean_model(**model_args)
    else:
        # create model args
        model_args = args_to_dict(args, mean_model_defaults().keys())
        save_args_dict(model_args, os.path.join(args.log_dir, "mean_model_args.pkl"))
        model = create_mean_model(**model_args)
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_train_data(
        args.data_dir,
        args.data_info_dict_path,
        args.batch_size,
        args.large_size,
        args.small_size,
        is_distributed=is_distributed,
    )

    logger.log("training...")
    SRMeanTrainLoop(
        model=model,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
        max_step=args.max_step,
        run_time=args.run_time,
        model_save_dir=args.model_save_dir,
        resume_checkpoint=args.resume_checkpoint,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        debug_mode=args.debug_mode,
    ).run_loop()

    logger.log("complete training.\n")


def create_argparser():
    # `defaults` contains args for SRMeanTrainLoop
    # As for model args, see sr_var.script_util.mean_model_defaults
    defaults = dict(
        # data
        data_dir="",
        data_info_dict_path="",
        # training
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # stop training
        max_step=100000000,
        run_time=-1.,
        # log and save
        log_dir="",
        log_interval=10,
        save_interval=10000,
        # model path
        model_save_dir="",
        resume_checkpoint="",  # if not "", it should be a file name.
        # debug
        debug_mode=False,
        # for distributed device, can be delete?
        local_rank=0,
    )
    # add model args to `defaults` dict
    defaults.update(mean_model_defaults())
    parser = argparse.ArgumentParser()
    # create parser args by defaults
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
