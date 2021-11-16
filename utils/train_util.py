"""
Basic training module for theorem 1 verification tasks
"""

import functools
import os
import blobfile as bf
import time
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .debug_util import *

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


# ytxie: This is the main part.
# ytxie: The parameter `microbatch` is used to train the model with smaller batch_size and accumulate gradient.
# ytxie: We add parameter `max_step` to control the iteration steps and `run_time` to save the last model.
class TrainLoop:
    """
    This class contains the training details.
    """
    def __init__(
            self,
            *,
            model,
            data,  # ytxie: iterate output is (batch, label)
            batch_size,
            microbatch,  # ytxie: if we don't use microbatch, set it as 0 or a negative integer.
            lr,
            weight_decay=0.0,
            lr_anneal_steps=0,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
            max_step=100000000,
            run_time=23.8,  # ytxie: hours, if we don't use run_time to control, set it as 0 or a negative value.
            model_save_dir="",
            resume_checkpoint="",  # ytxie: resume_checkpoint file name
            log_interval=10,
            save_interval=10000,
            debug_mode=False,  # whether to show gpu usage.
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.max_step = max_step
        self.run_time = run_time
        self.resume_checkpoint = resume_checkpoint
        self.model_save_dir = model_save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.debug_mode = debug_mode
        self.save_last = True

        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.step = self.resume_step
        if self.debug_mode:
            logger.log(f"This model contains {count_parameters_in_M(self.model)}M parameters")

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            initial_lg_loss_scale=initial_lg_loss_scale
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                # ytxie: Maybe this parameters is used in pytorch1.7, However in 1.6 version,
                # ytxie: it seems to ought be removed.
                # find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.log(
                    "Warning!"
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    # ytxie: We use the simplest method to load model parameters.
    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            model_checkpoint = os.path.join(self.model_save_dir, self.resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {model_checkpoint}...")
                self.model.load_state_dict(
                    th.load(
                        model_checkpoint, map_location=dist_util.dev()
                    )
                )
        dist_util.sync_params(self.model.parameters())

    # ytxie: We use the simplest method to load optimizer state.
    def _load_optimizer_state(self):
        # ytxie: The format `{self.resume_step:06}` may need to be changed.
        opt_checkpoint = bf.join(
            self.model_save_dir, f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    # ytxie: This function wraps the whole training process.
    def run_loop(self):
        start_time = time.time()
        # ytxie: When lr_anneal_steps > 0, it seems like maximum steps.
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
        ):
            batch, label = next(self.data)
            self.run_step(batch, label)
            self.step += 1

            if self.debug_mode and self.step % self.log_interval == 0:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())

            # save last model
            if self.run_time > 0 and time.time() - start_time > self.run_time * 3600 and self.save_last:
                self.save()
                self.save_last = False

            if self.step % self.log_interval == 0:
                logger.write_kv(self.step)
                logger.clear_kv()
            if self.step % self.save_interval == 0:
                self.save()
                # ytxie: How this code works? If this condition is not satisfies and
                # ytxie: `lr_anneal_steps` is 0, `run_loop` will continue running.
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if self.step >= self.max_step:
                break
        # Save the last checkpoint if it wasn't already saved.
        if self.step % self.save_interval != 0:
            self.save()

    def run_step(self, batch, label):
        self.forward_backward(batch, label)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()

    def forward_backward(self, batch, label):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro_input = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_label = label[i: i + self.microbatch].to(dist_util.dev())
            last_batch = (i + self.microbatch) >= batch.shape[0]
            micro_output = self.ddp_model(micro_input)

            compute_loss = functools.partial(
                th.nn.functional.mse_loss,
                micro_output,
                micro_label,
            )

            if last_batch or not self.use_ddp:
                loss = compute_loss()
            else:
                with self.ddp_model.no_sync():
                    loss = compute_loss()

            logger.log_kv("loss", loss)
            self.mp_trainer.backward(loss)

            self._post_process(micro_input, micro_label, micro_output, i)

    def _post_process(self, micro_input, micro_label, micro_output, i):
        """
        This function should be reloaded.
        :param micro_input:
        :param micro_label:
        :param micro_output:
        :param i:
        :return:
        """
        pass

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        # ytxie: In training process, step + resume_step will not be larger than lr_annela_steps.
        frac_done = self.step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def save(self):
        """
        Save the model and the optimizer state.
        """
        if dist.get_rank() == 0:
            state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
            logger.log(f"saving model at step {self.step}...")
            filename = f"model{self.step:06d}.pt"
            with bf.BlobFile(bf.join(self.model_save_dir, filename), "wb") as f:
                th.save(state_dict, f)

            with bf.BlobFile(
                bf.join(self.model_save_dir, f"opt{self.step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


# ytxie: We keep the filename form.
def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
