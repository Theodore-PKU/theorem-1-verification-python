"""
Testing process for super-resolution mean estimation
"""

import argparse
import os

from utils import dist_util, logger
from utils.script_util import add_dict_to_argparser, load_args_dict
from sr_var.mean_utils import SRMeanTestLoop, load_test_data
from sr_var.script_util import mean_model_defaults, create_mean_model


def main():
    args = create_argparser().parse_args()

    # distributed machines configuration
    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.log_dir, rank, is_distributed, is_write=True)
    logger.log("")
    logger.log("theorem 1 verification: super-resolution mean estimation training")
    logger.log("making device configuration...")  # pretend to make device configuration now :)

    logger.log("creating model...")
    model_args = load_args_dict(os.path.join(args.log_dir, "mean_model_args.pkl"))
    model = create_mean_model(**model_args)

    logger.log("creating data loader...")
    data = load_test_data(
        args.data_dir,
        args.data_info_dict_path,
        args.batch_size,
        args.large_size,
        args.small_size,
        is_distributed=is_distributed,
    )

    logger.log("testing...")
    SRMeanTestLoop(
        model=model,
        data=data,
        batch_size=args.batch_size,
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_fp16=args.use_fp16,
        debug_mode=args.debug_mode,
    ).run_loop()

    logger.log("complete testing.\n")


def create_argparser():
    # `defaults` contains args for SRMeanTestLoop
    # As for model args, see sr_var.script_util.mean_model_defaults
    defaults = dict(
        # data
        data_dir="",
        data_info_dict_path="",
        # testing
        batch_size=1,
        use_fp16=False,
        # log and save
        log_dir="",
        output_dir="",
        # model path
        model_path="",
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
