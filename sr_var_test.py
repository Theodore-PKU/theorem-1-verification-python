"""
Testing process for super-resolution var estimation
"""

import argparse
import os
import torch as th

from utils import dist_util, logger
from utils.script_util import add_dict_to_argparser, load_args_dict
from sr_var.var_utils import SRVarTestLoop, load_test_data
from sr_var.script_util import create_mean_model, var_model_defaults, create_var_model


def main():
    args = create_argparser().parse_args()

    # distributed machines configuration
    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.save, rank, is_distributed, is_write=True)
    logger.log("")
    logger.log("theorem 1 verification: super-resolution var estimation testing")
    logger.log("making device configuration...")  # pretend to make device configuration now :)

    logger.log("creating model...")
    model_args = load_args_dict(os.path.join(args.log_dir, "var_model_args.pkl"))
    model = create_var_model(**model_args)

    logger.log("load mean model...")
    mean_model_args = load_args_dict(args.mean_model_args_path)
    mean_model = create_mean_model(**mean_model_args)
    mean_model.load_state_dict(
        th.load(
            args.mean_model_path, map_location=dist_util.dev()
        )
    )
    mean_model.to(dist_util.dev())
    if mean_model_args['use_fp16']:
        mean_model.convert_to_fp16()

    logger.log("creating data loader...")
    data = load_test_data(
        args.data_dir,
        args.data_info_dict_path,
        args.batch_size,
        args.large_size,
        args.small_size,
        mean_model=mean_model,
        is_distributed=is_distributed,
    )

    logger.log("testing...")
    SRVarTestLoop(
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
    # `defaults` contains args for SRVarTestLoop
    # As for model args, see sr_var.script_util.var_model_defaults
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
        mean_model_path="",
        mean_model_args_path="",
        # debug
        debug_mode=False,
        # for distributed device, can be delete?
        local_rank=0,
    )
    # add model args to `defaults` dict
    defaults.update(var_model_defaults())
    parser = argparse.ArgumentParser()
    # create parser args by defaults
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
