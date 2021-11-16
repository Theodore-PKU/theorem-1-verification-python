"""
Basic testing module for theorem 1 verification tasks
"""

import os
import torch as th
import torch.distributed as dist

from . import dist_util, logger
from .debug_util import *
from .train_util import parse_resume_step_from_filename


class TestLoop:
    def __init__(
            self,
            *,
            model,
            data,  # ytxie: iterate output is (batch, batch_kwargs)
            batch_size,
            model_path="",  # ytxie: file path of model to load
            output_dir="",
            use_fp16=False,
            debug_mode=False,
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.resume_checkpoint = model_path
        self.output_dir = output_dir
        self.use_fp16 = use_fp16
        self.debug_mode = debug_mode

        self.global_batch = self.batch_size * dist.get_world_size()  # seems to be useless.

        self.resume_step = 0  # it will be changed in self._load_parameters(). We use it to indicate which model.
        self._load_parameters()
        if self.debug_mode:
            logger.log(f"This model contains {count_parameters_in_M(self.model)}M parameters")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.step = 0

    # ytxie: We use the simplest method to load model parameters.
    def _load_parameters(self):
        self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
        logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
        self.model.load_state_dict(
            th.load(
                self.resume_checkpoint,
                map_location="cpu",
            )
        )
        self.model.to(dist_util.dev())
        if self.use_fp16:
            self.model.convert_to_fp16()

        self.model.eval()

    # ytxie: This function wraps the whole training process.
    def run_loop(self):
        for batch, batch_kwargs in self.data:
            self.forward_backward(batch, batch_kwargs)
            self.step += 1
            if self.debug_mode:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())
            if self.step % 10 == 0:
                logger.log(f"have run {self.step} steps")

        dist.barrier()

    def forward_backward(self, batch, batch_kwargs):
        batch = batch.to(dist_util.dev())
        with th.no_grad():
            output = self.model(batch)

        self._post_process(batch, output, batch_kwargs)

    def _post_process(self, batch, output, batch_kwargs):
        pass
