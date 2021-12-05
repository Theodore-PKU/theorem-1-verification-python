"""
For simple training of MNIST classification task.
"""

import argparse
import os

from utils import dist_util, logger
from utils.script_util import add_dict_to_argparser, args_to_dict, load_args_dict, save_args_dict
from classification.utils import (
    create_model, model_defaults, load_train_data, load_test_data, TrainLoop,
    PredictAnalysisLoop, load_ordered_data, GenNoisyLabelLoop, NoiseCETrainLoop,
)


def main():
    args = create_argparser().parse_args()

    # distributed machines configuration
    # is_distributed, rank = dist_util.setup_dist()
    is_distributed = False
    logger.configure(args.log_dir, 0, is_distributed, is_write=True)
    logger.log("")
    logger.log("theorem 1 verification: mnist classification")
    logger.log("making device configuration...")  # pretend to make device configuration now :)

    logger.log("creating model...")
    if args.resume_checkpoint:
        # load model args which has been created last time
        model_args = load_args_dict(os.path.join(args.log_dir, "model_args.pkl"))
        model = create_model(**model_args)
    else:
        # create model args
        model_args = args_to_dict(args, model_defaults().keys())
        save_args_dict(model_args, os.path.join(args.log_dir, "model_args.pkl"))
        model = create_model(**model_args)
    model.to(dist_util.dev())

    if args.task == "train":
        logger.log("creating data loader...")
        train_data = load_train_data(
            args.data_dir,
            args.batch_size,
            noise_label_file=None,
            is_distributed=is_distributed,
            is_train=True,
        )

        test_data_for_train_set = load_test_data(
            args.data_dir,
            args.batch_size,
            noise_label_file=None,
            is_distributed=is_distributed,
            is_train=True,
        )

        test_data_for_test_set = load_test_data(
            args.data_dir,
            args.batch_size,
            noise_label_file=None,
            is_distributed=is_distributed,
            is_train=False,
        )

        logger.log("training...")
        TrainLoop(
            model=model,
            train_data=train_data,
            test_data_for_train_set=test_data_for_train_set,
            test_data_for_test_set=test_data_for_test_set,
            lr=args.lr,
            max_step=args.max_step,
            model_save_dir=args.model_save_dir,
            resume_checkpoint=args.resume_checkpoint,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
        ).run_loop()

        logger.log("complete training.\n")

    elif args.task == "predict_analysis":
        logger.log("creating data loader...")
        data = load_test_data(
            args.data_dir,
            args.batch_size,
            noise_label_file=args.noise_label_file,
            is_train=True,
        )

        logger.log("analysing prediction result...")
        PredictAnalysisLoop(
            model=model,
            test_data=data,
            noise_type=args.noise_type,
            model_save_dir=args.model_save_dir,
            resume_checkpoint=args.resume_checkpoint,
        ).run_loop()

        logger.log("complete analysing.\n")

    elif args.task == "generate_noisy_label":
        if args.noise_type != "model":
            model = None
        logger.log("creating data loader...")
        data = load_ordered_data(
            args.data_dir,
            args.batch_size,
            is_train=True
        )

        logger.log("generating noisy label...")

        GenNoisyLabelLoop(
            alpha=args.alpha,
            noise_type=args.noise_type,
            model=model,
            data=data,
            model_save_dir=args.model_save_dir,
            resume_checkpoint=args.resume_checkpoint,
        ).run_loop()

        logger.log("complete generating.\n")

    elif args.task == "train_on_noisy_label":
        logger.log(f"creating data loader from noisy label dataset {args.noise_label_file}...")
        train_data = load_train_data(
            args.data_dir,
            args.batch_size,
            noise_label_file=args.noise_label_file,
            is_distributed=False,
            is_train=True,
        )

        test_data_for_train_set = load_test_data(
            args.data_dir,
            args.batch_size,
            noise_label_file=args.noise_label_file,
            is_distributed=is_distributed,
            is_train=True
        )

        test_data_for_test_set = load_test_data(
            args.data_dir,
            args.batch_size,
            noise_label_file=None,
            is_distributed=is_distributed,
            is_train=False
        )

        logger.log(f"training on noisy label...")
        NoiseCETrainLoop(
            model=model,
            train_data=train_data,
            test_data_for_test_set=test_data_for_test_set,
            test_data_for_train_set=test_data_for_train_set,
            lr=args.lr,
            max_step=args.max_step,
            model_save_dir=args.model_save_dir,
            resume_checkpoint=args.resume_checkpoint,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
        ).run_loop()

        logger.log("complete training.\n")


def create_argparser():
    # `defaults` contains args for SRMeanTrainLoop
    # As for model args, see sr_var.script_util.mean_model_defaults
    defaults = dict(
        # data
        data_dir="data/mnist",
        # training
        lr=1e-4,
        batch_size=128,
        # stop training
        max_step=20000,
        # log and save
        log_dir="logs/mnist_classification",
        log_interval=10,
        save_interval=2000,
        # model path
        model_save_dir="",
        resume_checkpoint="",
        # task, "train" or "predict_analysis" or
        # "generate_noisy_label" or "train_on_noisy_label"
        task="predict_analysis",
        alpha=0.5,  # only used when task is "generate_noisy_label"
        noise_type="model",  # only used when task is "generate_noisy_label" or "predict_analysis"
        noise_label_file="model_noisy_label_0.5.pkl"
        # only used when task is "predict_analysis" or "train_on_noisy_label"
    )
    # add model args to `defaults` dict
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    # create parser args by defaults
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
