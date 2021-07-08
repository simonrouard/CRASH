import numpy as np
import os
import re
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from glob import glob

from dataset import from_path as dataset_from_path
from model import UNet
from getters import get_sde


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class Learner:
    def __init__(
        self, model_dir, model, ema_weights, sde, train_set, test_set, optimizer, params
    ):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.ema_weights = ema_weights
        self.sde = get_sde(params['sde_type'], params['sde_kwargs'])
        self.ema_rate = params['ema_rate']
        self.train_set = train_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.params = params
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=params['scheduler_step_size'], gamma=params['scheduler_gamma']
        )

        self.step = 0
        self.is_master = True

        self.loss_fn = nn.MSELoss()
        self.v_loss = nn.MSELoss(reduction="none")
        self.summary_writer = None
        self.n_bins = params['n_bins']
        self.num_elems_in_bins_train = np.zeros(self.n_bins)
        self.sum_loss_in_bins_train = np.zeros(self.n_bins)
        self.num_elems_in_bins_test = np.zeros(self.n_bins)
        self.sum_loss_in_bins_test = np.zeros(self.n_bins)
        self.cum_grad_norms = 0

    def state_dict(self):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "step": self.step,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            'ema_weights': [elem.cpu() for elem in self.ema_weights],
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.step = state_dict["step"]
        self.ema_weights = state_dict['ema_weights']

    def save_to_checkpoint(self, filename="weights"):
        save_basename = f"{filename}-{self.step}.pt"
        save_name = f"{self.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)

    def restore_from_checkpoint(self, checkpoint_id=None):
        try:
            if checkpoint_id is None:
                # find latest checkpoint_id
                list_weights = glob(f'{self.model_dir}/weights-*')
                id_regex = re.compile('weights-(\d*)')
                list_ids = [int(id_regex.search(weight_path).groups()[0])
                            for weight_path in list_weights]
                checkpoint_id = max(list_ids)

            checkpoint = torch.load(
                f"{self.model_dir}/weights-{checkpoint_id}.pt")
            self.load_state_dict(checkpoint)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def train(self):
        device = next(self.model.parameters()).device
        while True:
            for features in (
                tqdm(self.train_set,
                     desc=f"Epoch {self.step // len(self.train_set)}")
                if self.is_master
                else self.train_set
            ):
                features = _nested_map(
                    features,
                    lambda x: x.to(device) if isinstance(
                        x, torch.Tensor) else x,
                )
                loss = self.train_step(features)
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f"Detected NaN loss at step {self.step}.")
                if self.is_master:
                    if self.step % 250 == 249:
                        self._write_summary(self.step)

                    if self.step % self.params['num_steps_to_test'] == self.params['num_steps_to_test'] - 1:
                        self.test_set_evaluation()
                        self._write_test_summary(self.step)

                    if self.step % (self.params['num_epochs_to_save'] * len(self.train_set)) == 10:
                        self.save_to_checkpoint()
                self.step += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        audio = features["audio"]

        N, T = audio.shape

        t = torch.rand(N, 1, device=audio.device)
        noise = torch.randn_like(audio)
        noisy_audio = self.sde.perturb(audio, t, noise)
        sigma = self.sde.sigma(t)
        predicted = self.model(noisy_audio, sigma)
        loss = self.loss_fn(noise, predicted)

        loss.backward()
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        print(len(self.ema_weights))
        self.update_ema_weights()

        t_detach = t.clone().detach().cpu().numpy()
        t_detach = np.reshape(t_detach, -1)

        vectorial_loss = self.v_loss(noise, predicted).detach()

        vectorial_loss = torch.mean(vectorial_loss, 1).cpu().numpy()
        vectorial_loss = np.reshape(vectorial_loss, -1)

        self.update_conditioned_loss(vectorial_loss, t_detach, True)

        self.cum_grad_norms += self.grad_norm

        return loss

    def _write_summary(self, step):
        loss_in_bins_train = np.divide(
            self.sum_loss_in_bins_train, self.num_elems_in_bins_train
        )
        dic_loss_train = {}
        for k in range(self.n_bins):
            dic_loss_train["loss_bin_" + str(k)] = loss_in_bins_train[k]

        sum_loss_n_steps = np.sum(self.sum_loss_in_bins_train)
        mean_grad_norms = self.cum_grad_norms / self.num_elems_in_bins_train.sum() * \
            self.params['batch_size']
        writer = self.summary_writer or SummaryWriter(
            self.model_dir, purge_step=step)

        writer.add_scalar('train/sum_loss_on_n_steps',
                          sum_loss_n_steps, step)
        writer.add_scalar("train/mean_grad_norm", mean_grad_norms, step)
        writer.add_scalars("train/conditioned_loss", dic_loss_train, step)
        writer.flush()
        self.summary_writer = writer
        self.num_elems_in_bins_train = np.zeros(self.n_bins)
        self.sum_loss_in_bins_train = np.zeros(self.n_bins)
        self.cum_grad_norms = 0

    def _write_test_summary(self, step):
        # Same thing for test set
        loss_in_bins_test = np.divide(
            self.sum_loss_in_bins_test, self.num_elems_in_bins_test
        )
        dic_loss_test = {}
        for k in range(self.n_bins):
            dic_loss_test["loss_bin_" + str(k)] = loss_in_bins_test[k]

        writer = self.summary_writer or SummaryWriter(
            self.model_dir, purge_step=step)
        writer.add_scalars("test/conditioned_loss", dic_loss_test, step)
        writer.flush()
        self.summary_writer = writer
        self.num_elems_in_bins_test = np.zeros(self.n_bins)
        self.sum_loss_in_bins_test = np.zeros(self.n_bins)

    def update_conditioned_loss(self, vectorial_loss, continuous_array, isTrain):
        continuous_array = np.trunc(self.n_bins * continuous_array)
        continuous_array = continuous_array.astype(int)
        if isTrain:
            for k in range(len(continuous_array)):
                self.num_elems_in_bins_train[continuous_array[k]] += 1
                self.sum_loss_in_bins_train[continuous_array[k]
                                            ] += vectorial_loss[k]
        else:
            for k in range(len(continuous_array)):
                self.num_elems_in_bins_test[continuous_array[k]] += 1
                self.sum_loss_in_bins_test[continuous_array[k]
                                           ] += vectorial_loss[k]

    def update_ema_weights(self):
        for ema_param, param in zip(self.ema_weights, self.model.parameters()):
            if param.requires_grad:
                ema_param *= self.ema_rate * ema_param
                ema_param += (1. - self.ema_rate) * param.detach()

    def test_set_evaluation(self):
        with torch.no_grad():
            self.model.eval()
            for features in self.test_set:
                audio = features["audio"].cuda()

                N, T = audio.shape

                t = torch.rand(N, 1, device=audio.device)
                noise = torch.randn_like(audio)
                noisy_audio = self.sde.perturb(audio, t, noise)
                sigma = self.sde.sigma(t)
                predicted = self.model(noisy_audio, sigma)

                vectorial_loss = self.v_loss(noise, predicted).detach()

                vectorial_loss = torch.mean(vectorial_loss, 1).cpu().numpy()
                vectorial_loss = np.reshape(vectorial_loss, -1)
                t = t.cpu().numpy()
                t = np.reshape(t, -1)
                self.update_conditioned_loss(
                    vectorial_loss, t, False)


def _train_impl(replica_id, model, ema_weights, sde, train_set, test_set, params):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    learner = Learner(
        params['model_dir'], model, ema_weights, sde, train_set, test_set, opt, params
    )
    learner.is_master = replica_id == 0
    learner.restore_from_checkpoint(params['checkpoint_id'])
    learner.train()


def train(params):
    model = UNet().cuda()
    ema_weights = [param.clone().detach() for param in model.parameters()]
    print(len(ema_weights))
    sde = get_sde(params['sde_type'], params['sde_kwargs'])
    train_set = dataset_from_path(params['train_dirs'], params)
    test_set = dataset_from_path(params['test_dirs'], params)

    _train_impl(0, model, ema_weights, sde, train_set, test_set, params)


def train_distributed(replica_id, replica_count, port, params):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        "nccl", rank=replica_id, world_size=replica_count
    )
    device = torch.device("cuda", replica_id)
    torch.cuda.set_device(device)

    model = UNet().to(device)
    ema_weights = [param.clone().detach() for param in model.parameters()]
    print(len(ema_weights))

    sde = get_sde(params['sde_type'], params['sde_kwargs'])
    train_set = dataset_from_path(
        params['train_dirs'], params, is_distributed=True)
    test_set = dataset_from_path(
        params['test_dirs'], params, is_distributed=True)

    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, ema_weights, sde,
                train_set, test_set, params)
