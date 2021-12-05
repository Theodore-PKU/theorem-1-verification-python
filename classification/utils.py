"""
Helpers for MNIST classification task.
"""

import os
import blobfile as bf

import torch as th
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

from utils import dist_util, logger
from utils.train_util import parse_resume_step_from_filename
from models.sim_cnn import SimpleCNN
from .mnist_dataset import load_data, MNISTDataset


# ================================================================
# TrainLoop and TestLoop
# ================================================================

class TrainLoop:

    def __init__(
            self,
            model,
            train_data,
            test_data_for_test_set,
            test_data_for_train_set,
            lr,
            max_step,
            model_save_dir,
            resume_checkpoint="",
            log_interval=10,
            save_interval=1000,
            debug_mode=False,
    ):
        self.model = model
        self.train_data = train_data
        self.test_data_for_train_set = test_data_for_train_set
        self.test_data_for_test_set = test_data_for_test_set
        self.lr = lr
        self.max_step = max_step
        self.resume_checkpoint = resume_checkpoint
        self.model_save_dir = model_save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.debug_mode = debug_mode

        self.resume_step = 0
        self.sync_cuda = th.cuda.is_available()

        self._load_and_parameters()
        self.step = self.resume_step

        self.opt = Adam(
            self.model.parameters(), lr=self.lr
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

    # ytxie: We use the simplest method to load model parameters.
    def _load_and_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            model_checkpoint = os.path.join(self.model_save_dir, self.resume_checkpoint)
            logger.log(f"loading model from checkpoint: {model_checkpoint}...")
            self.model.load_state_dict(
                th.load(
                    model_checkpoint, map_location=dist_util.dev()
                )
            )

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
        self.model.train()
        while self.step < self.max_step:
            batch, clean_label, noisy_label, prob = next(self.train_data)
            self.run_step(batch, clean_label)
            self.step += 1

            if self.step % self.log_interval == 0:
                logger.write_kv(self.step)
                logger.clear_kv()
            if self.step % self.save_interval == 0:
                self.save_model()
                self.test_model()
                self.model.train()

        # Save the last checkpoint if it wasn't already saved.
        if self.step % self.save_interval != 0:
            self.save_model()
            self.test_model()

    def run_step(self, batch, label):
        loss = self.forward_backward(batch, label)
        loss.backward()
        self.opt.step()

    def forward_backward(self, batch, label):
        batch = batch.to(dist_util.dev())
        label = label.to(dist_util.dev())
        self.opt.zero_grad()
        output = self.model(batch)
        loss = F.cross_entropy(output, label)
        logger.log_kv("loss", loss)
        return loss

    def save_model(self):
        """
        Save the model and the optimizer state, and test the model.
        """
        state_dict = self.model.state_dict()
        logger.log(f"saving model at step {self.step}...")
        filename = f"model{self.step:06d}.pt"
        with bf.BlobFile(bf.join(self.model_save_dir, filename), "wb") as f:
            th.save(state_dict, f)

        with bf.BlobFile(
                bf.join(self.model_save_dir, f"opt{self.step:06d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def test_model(self):
        logger.log("test on training set")
        TestLoop(
            model=self.model,
            test_data=self.test_data_for_train_set,
            test_on_clean_label=True,
        ).run_loop()

        logger.log("test on testing set")
        TestLoop(
            model=self.model,
            test_data=self.test_data_for_test_set,
            test_on_clean_label=True,
        ).run_loop()


class TestLoop:

    def __init__(
            self,
            *,
            model,
            test_data,
            test_on_clean_label=True,
            model_save_dir="",
            resume_checkpoint=""
    ):
        self.model = model
        self.test_data = test_data
        self.test_on_clean_label = test_on_clean_label
        self.model_save_dir = model_save_dir
        self.resume_checkpoint = resume_checkpoint

        if self.resume_checkpoint:
            self._load_parameters()
        self.model.eval()

        self.step = 0
        self.accuracy = 0.
        self.num_data = 0.
        self.ce_loss = 0.

    def _load_parameters(self):
        self.resume_checkpoint = os.path.join(self.model_save_dir, self.resume_checkpoint)
        logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
        self.model.load_state_dict(
            th.load(
                self.resume_checkpoint,
                map_location="cpu",
            )
        )
        self.model.to(dist_util.dev())

    def run_loop(self):
        for batch, clean_label, noisy_label, prob in self.test_data:
            batch = batch.to(dist_util.dev())
            if self.test_on_clean_label:
                self.ce_loss += self.forward_backward(batch, clean_label)
            else:
                self.ce_loss += self.forward_backward(batch, noisy_label)
            if self.step % 100 == 0:
                logger.log(f"have run {self.step} steps")

        self.accuracy = self.accuracy / self.num_data
        self.ce_loss = self.ce_loss / self.step
        logger.log(f"accuracy: {self.accuracy:.4f}")
        logger.log(f"ce loss: {self.ce_loss:.4f}")

    def forward_backward(self, batch, label):
        batch = batch.to(dist_util.dev())
        label = label.to(dist_util.dev())
        with th.no_grad():
            output = self.model(batch)
        pred = th.argmax(output, dim=1)
        ce_loss = F.cross_entropy(output, label)
        self.accuracy += th.sum(pred == label)
        self.num_data += len(batch)
        self.step += 1
        return ce_loss


class GenNoisyLabelLoop:

    def __init__(
            self,
            alpha,
            noise_type,
            model,
            data,
            model_save_dir="",
            resume_checkpoint=""
    ):
        self.category_list = np.arange(10)
        self.alpha = alpha
        self.noise_type = noise_type
        self.new_label_list = []
        self.incorrect_max_prob_count = 0
        self.accuracy = 0.
        self.num_data = 0.

        self.model = model
        self.data = data
        self.model_save_dir = model_save_dir
        self.resume_checkpoint = resume_checkpoint

        if self.resume_checkpoint and model is not None:
            self._load_parameters()
            self.model.eval()

    def _load_parameters(self):
        self.resume_checkpoint = os.path.join(self.model_save_dir, self.resume_checkpoint)
        logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
        self.model.load_state_dict(
            th.load(
                self.resume_checkpoint,
                map_location="cpu",
            )
        )
        self.model.to(dist_util.dev())

    def run_loop(self):
        for batch, label in self.data:
            self.forward_backward(batch, label)

        self.accuracy = self.accuracy / self.num_data
        logger.log(f"accuracy of noisy label: {self.accuracy:.4f}")
        logger.log(f"max prob is incorrect category ratio: {self.incorrect_max_prob_count / self.num_data:.4f}")

        logger.log(f"save noisy label as a pickle file")
        with open(os.path.join(logger.get_dir(), f"{self.noise_type}_noisy_label_{self.alpha}.pkl"), "wb") as f:
            pickle.dump(self.new_label_list, f)

    def forward_backward(self, batch, label):
        pred_prob = None
        if self.noise_type == "model":
            batch = batch.to(dist_util.dev())
            label = label.to(dist_util.dev())
            with th.no_grad():
                output = self.model(batch)

            pred_prob = F.softmax(output, dim=1).detach().cpu().numpy()

        for i in range(len(label)):
            category = int(label[i])

            if self.noise_type == "model":
                new_prob = np.copy(pred_prob[i])
                right_prob = new_prob[category]
                beta = (1. - right_prob * self.alpha) / (1. - right_prob + 1e-6)
                new_prob = new_prob * beta
                new_prob[category] = right_prob * self.alpha

            elif self.noise_type == "uniform":
                new_prob = np.ones(10)
                new_prob = new_prob * (1 - self.alpha) / 9.
                new_prob[category] = self.alpha

            elif self.noise_type == "fix":
                new_prob = np.zeros(10)
                new_prob[category] = self.alpha
                new_prob[(category + 1) % 10] = 1 - self.alpha

            else:
                raise ValueError

            new_prob = new_prob / np.sum(new_prob)
            new_category = int(np.random.choice(self.category_list, 1, p=new_prob))

            self.new_label_list.append((new_category, new_prob))
            if new_category == category:
                self.accuracy += 1
            if np.argmax(new_prob) != category:
                self.incorrect_max_prob_count += 1

        self.num_data += len(batch)


class NoiseCETrainLoop(TrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_loop(self):
        self.model.train()
        while self.step < self.max_step:
            batch, clean_label, noisy_label, prob = next(self.train_data)

            self.run_step(batch, noisy_label)
            self.step += 1

            if self.step % self.log_interval == 0:
                logger.write_kv(self.step)
                logger.clear_kv()
            if self.step % self.save_interval == 0:
                self.save_model()
                self.test_model()
                self.model.train()

        # Save the last checkpoint if it wasn't already saved.
        if self.step % self.save_interval != 0:
            self.save_model()

    def test_model(self):
        logger.log("test on training set to compute average ce loss when training on noisy label")
        TestLoop(
            model=self.model,
            test_data=self.test_data_for_train_set,
            test_on_clean_label=False,
        ).run_loop()

        logger.log("test on training set to compute classification accuracy on clean training data")
        TestLoop(
            model=self.model,
            test_data=self.test_data_for_train_set,
            test_on_clean_label=True,
        ).run_loop()

        logger.log("test on testing set to compute classification accuracy on test data")
        TestLoop(
            model=self.model,
            test_data=self.test_data_for_test_set,
            test_on_clean_label=True,
        ).run_loop()


class PredictAnalysisLoop(TestLoop):

    def __init__(self, noise_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category_predict_prob = np.zeros((10, 10))
        self.category_predict = np.zeros(10, dtype=np.int)
        self.category_wrong_predict_prob = np.zeros((10, 10))
        self.category_wrong_predict = np.zeros((10, 10), dtype=np.int)
        self.category_noise_prob = np.zeros((10, 10))
        self.pred_wrong_count = 0
        self.pred_right_count = 0
        self.max_figures = 200
        self.noise_type = noise_type

    def run_loop(self):
        plt.set_loglevel('WARNING')
        for batch, clean_label, noisy_label, prob in self.test_data:
            self.forward_backward(batch, clean_label, noisy_label, prob)
            if self.step % 100 == 0:
                logger.log(f"have run {self.step} steps")

        self.accuracy = self.accuracy / self.num_data
        logger.log(f"accuracy : {self.accuracy:.4f}")

        self.print_predict_analysis()

    def forward_backward(self, batch, clean_label, noisy_label, prob):
        self.step += 1
        batch = batch.to(dist_util.dev())
        clean_label = clean_label.to(dist_util.dev())
        with th.no_grad():
            output = self.model(batch)
        pred = th.argmax(output, dim=1)
        self.accuracy += th.sum(pred == clean_label)
        self.num_data += len(batch)

        prob = prob.detach().cpu().numpy()
        pred_prob = F.softmax(output, dim=1).detach().cpu().numpy()
        for i in range(len(clean_label)):
            category = int(clean_label[i])
            category_pred = int(pred[i])
            self.category_predict[category] += 1
            self.category_predict_prob[category] += pred_prob[i]
            self.category_noise_prob[category] += prob[i]
            if category_pred != category:
                self.pred_wrong_count += 1
                self.category_wrong_predict[category][category_pred] += 1
                self.category_wrong_predict_prob[category] += pred_prob[i]
                if self.pred_wrong_count <= self.max_figures:
                    self.save_wrong_predict(batch[i], category, category_pred, prob[i], pred_prob[i])
            else:
                self.pred_right_count += 1
                if self.pred_right_count <= self.max_figures:
                    self.save_right_predict(batch[i], category, category_pred, prob[i], pred_prob[i])

    def print_predict_analysis(self):
        wrong_predict_count = np.sum(self.category_wrong_predict)
        print((self.num_data - wrong_predict_count) / self.num_data)

        # compute averagey predict prob
        self.category_predict_prob = self.category_predict_prob / self.category_predict.reshape(10, 1)
        self.category_wrong_predict_prob = self.category_wrong_predict_prob / \
                                           np.sum(self.category_wrong_predict, axis=1).reshape(10, 1)
        self.category_noise_prob = self.category_noise_prob / self.category_predict.reshape(10, 1)

        for i in range(10):
            x_data = [str(j) for j in range(10)]
            if self.noise_type == "":
                fig, ax1 = plt.subplots()
                ax1.set_ylabel("prob")
                ax1.set_xlabel("category")
                l1 = ax1.plot(x_data, self.category_predict_prob[i], color="blue")
                l2 = ax1.plot(x_data, self.category_wrong_predict_prob[i], color="purple")
                ax2 = ax1.twinx()
                ax2.set_ylabel("number")
                l3 = ax2.plot(x_data, self.category_wrong_predict[i], color="green")
                plt.legend(l1 + l2 + l3, ["avg pred prob", "avg pred prob when incorrect", "count for incorrect pred"])
                plt.title(f"prediction for category {i}")
                plt.savefig("temp.png")
            else:
                fig, ax1 = plt.subplots()
                ax1.set_ylabel("prob")
                ax1.set_xlabel("category")
                l1 = ax1.plot(x_data, self.category_predict_prob[i], color="blue")
                l2 = ax1.plot(x_data, self.category_noise_prob[i], color="purple")
                ax2 = ax1.twinx()
                ax2.set_ylabel("number")
                l3 = ax2.plot(x_data, self.category_wrong_predict[i], color="green")
                plt.legend(l1 + l2 + l3, ["avg pred prob", "avg sample prob", "count for incorrect pred"])
                plt.title(f"prediction for category {i} with noise of {self.noise_type}")
                plt.savefig("temp.png")
            self._save_image_to_tensorboard("predict_analysis", i)

    def save_wrong_predict(self, image, category, category_pred, prob, pred_prob):
        x_data = [str(i) for i in range(10)]
        plt.figure()
        plt.plot(x_data, pred_prob)
        plt.plot(x_data, prob)
        plt.ylabel("probability")
        plt.xlabel("category")
        plt.title(f"label: {category}, pred: {category_pred}")
        plt.legend(["pred prob", "sample prob"])
        plt.savefig("temp.png")
        plt.close()
        self._save_image_to_tensorboard("wrong_pred_prob", self.pred_wrong_count)
        logger.get_current().write_image(
            'input_of_wrong_pred',
            image,
            self.pred_wrong_count
        )

    def save_right_predict(self, image, category, category_pred, prob, pred_prob):
        x_data = [str(i) for i in range(10)]
        plt.figure()
        plt.plot(x_data, pred_prob)
        plt.plot(x_data, prob)
        plt.ylabel("probability")
        plt.xlabel("category")
        plt.title(f"label: {category}, pred: {category_pred}")
        plt.legend(["pred prob", "sample prob"])
        plt.savefig("temp.png")
        plt.close()
        self._save_image_to_tensorboard("right_pred_prob", self.pred_right_count)
        logger.get_current().write_image(
            'input_of_right_pred',
            image,
            self.pred_right_count
        )

    def _save_image_to_tensorboard(self, name, step):
        with open("temp.png", "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        th_image = th.Tensor(np.array(pil_image)).permute(2, 0, 1) / 255.
        logger.get_current().write_image(
            name,
            th_image,
            step,
        )
        os.remove("temp.png")


# ================================================================
# Dataset
# ================================================================

def load_train_data(
        data_dir,
        batch_size,
        noise_label_file,
        is_distributed=False,
        is_train=True,
):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        noise_label_file=noise_label_file,
        is_distributed=is_distributed,
        is_train=is_train,
        to_train=True,
    )
    while True:
        yield from data


def load_test_data(
        data_dir,
        batch_size,
        noise_label_file,
        is_distributed=False,
        is_train=False,
):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        noise_label_file=noise_label_file,
        is_distributed=is_distributed,
        is_train=is_train,
        to_train=False,
    )
    return data


def load_ordered_data(data_dir, batch_size, is_train=True):
    dataset = MNISTDataset(data_dir, noise_label_file=None, train=is_train)
    num_data = len(dataset)
    index = 0
    while index < num_data:
        index_ = min(index + batch_size, num_data)
        batch = []
        label = []
        for i in range(index, index_):
            image, clean_label, noisy_label, prob = dataset[i]
            batch.append(image)
            label.append(clean_label)
        batch = th.stack(batch)
        label = th.stack(label)
        index = index_
        yield batch, label


def create_model(
        input_dim,
        hidden_dims,
):
    return SimpleCNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
    )


def model_defaults():
    return dict(
        input_dim=256,
        hidden_dims=[128, 64]
    )
