import logging
import os
from argparse import Namespace

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.datasets import DatasetFolder
from pytorch_lightning.core import LightningModule

import utils.helpers as H
from data.lidc import LIDCNodulesDataset
from models.rls import RLSModule, init_levelset


log = logging.getLogger("lightning_boilerplates.crls")


class CRLSModel(LightningModule):
    def __init__(self, config: Namespace):
        super().__init__()
        self.metaconf = config.metaconf
        self.hparams = Namespace(**config.hyperparams)
        self.dataset_params = Namespace(**config.dataloaders["train"])

        inp_image_size = [self.dataset_params.params["cube_voxelsize"]] * 2
        self.rls_model = RLSModule(inp_image_size)

    def forward(self, input, hidden, writer: SummaryWriter = None, step: int = None):
        return self.rls_model(input, hidden, writer, step)

    def configure_optimizers(self):
        lr = self.hparams.lr
        alpha = self.hparams.alpha
        optimizer = torch.optim.RMSprop(self.rls_model.parameters(), lr=lr, alpha=alpha)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.lr_shed_rate),
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def optimizer_step(
        self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None
    ):
        g_norm = H.get_gradient_norm(self.parameters())
        gradplot_savepath = H.makedirs(
            os.path.join(self.metaconf["ws_path"], "artifacts", "gradflow_plots")
        )
        fig = H.plot_grad_flow(self.named_parameters(), "RLS", 0, gradplot_savepath)
        self.logger.experiment.add_figure("gradflow_plots", fig, self.global_step)
        if np.isnan(g_norm):
            log.warning("  gradient norm is NaN -> skip")
            optimizer.zero_grad()
            return
        elif g_norm > self.hparams.optimizer_max_grad_norm:
            log.warning(f"  gradient norm is too high: {g_norm:.5f} -> clip to OPTIMIZER_MAX_GRAD_NORM")
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.optimizer_max_grad_norm)
        else:
            log.info(f"  gradient norm: {g_norm:.5f}")
        optimizer.step()
        optimizer.zero_grad()

    def prepare_data(self):
        """Prepare and save dataset as TensorDataset to improve training speed.
        """
        self.generic_dataset = LIDCNodulesDataset(**self.dataset_params.params)
        log.info(f"DATASET SIZE: {len(self.generic_dataset)}")

        tensor_dataset_path = self.__prepare_tensor_dataset()

        self.dataset = DatasetFolder(tensor_dataset_path, torch.load, ("pt"))
        self.dataset.norm = self.generic_dataset.norm

        train_inds, val_inds, test_inds = H.train_val_holdout_split(self.dataset)
        self.train_sampler = SubsetRandomSampler(train_inds)
        self.val_sampler = SubsetRandomSampler(val_inds)
        self.test_subset = Subset(self.dataset, test_inds)

    def train_dataloader(self):
        dl = DataLoader(
            self.dataset,
            sampler=self.train_sampler,
            batch_size=self.hparams.batch_size,
            num_workers=self.metaconf["dl_workers"],
        )
        return dl

    def val_dataloader(self):
        dl = DataLoader(
            self.dataset,
            sampler=self.val_sampler,
            batch_size=self.hparams.batch_size,
            num_workers=self.metaconf["dl_workers"],
        )
        return dl

    # def test_dataloader(self):
    #     dl = DataLoader(self.test_subset)
    #     return dl

    def loss_f(self, y_hat, y):
        return F.binary_cross_entropy(torch.sigmoid(y_hat), y.float())

    def training_step(self, batch, batch_idx):
        nodules, masks = batch[0]["nodule"], batch[0]["mask"]
        nodules, masks = nodules[:, :, nodules.size(2) // 2, :, :], masks[:, masks.size(2) // 2, :, :]

        hiddens = init_levelset(nodules.shape[-2:], shape=self.hparams.levelset_init)
        hiddens = hiddens.repeat(nodules.size(0), 1, 1, 1).type_as(nodules)
        for t in range(self.hparams.num_T):
            step = self.current_epoch * 1000 + batch_idx * 100 + t
            outputs, hiddens = self.forward(nodules, hiddens, self.logger.experiment, step=step)

        loss = self.loss_f(outputs, masks)

        tqdm_dict = {"train_loss": loss}
        output = {
            "batch": batch[0],
            "loss": loss,
            "log": tqdm_dict,
        }
        return output

    def training_step_end(self, output):
        if self.global_step % 20 == 0:
            imgs, masks = output["batch"]["nodule"], output["batch"]["mask"]
            imgs, masks = imgs[:, :, imgs.size(2) // 2, :, :], masks[:, masks.size(2) // 2, :, :]
            imgs_in_hu = self.dataset.norm.denorm(imgs)
            grid = torchvision.utils.make_grid(
                imgs_in_hu,
                nrow=4,
                normalize=True,
                range=(
                    self.dataset_params.params["ct_clip_range"][0],
                    self.dataset_params.params["ct_clip_range"][1],
                ),
            )
            mask_grid = torchvision.utils.make_grid(masks.unsqueeze(1), 4)
            self.logger.experiment.add_image(f"input/images", grid, self.global_step)
            self.logger.experiment.add_image(f"input/masks", mask_grid, self.global_step)
        del output["batch"]
        return output

    def validation_step(self, batch, batch_idx):
        nodules, masks = batch[0]["nodule"], batch[0]["mask"]
        nodules, masks = nodules[:, :, nodules.size(2) // 2, :, :], masks[:, masks.size(2) // 2, :, :]

        hiddens = init_levelset(nodules.shape[-2:], shape=self.hparams.levelset_init)
        hiddens = hiddens.repeat(nodules.size(0), 1, 1, 1).type_as(nodules)
        for t in range(self.hparams.num_T):
            outputs, hiddens = self.forward(nodules, hiddens)

        if self.global_step % 60 == 0:
            imgs_in_hu = self.dataset.norm.denorm(nodules)
            grid = torchvision.utils.make_grid(
                imgs_in_hu,
                nrow=4,
                normalize=True,
                range=(
                    self.dataset_params.params["ct_clip_range"][0],
                    self.dataset_params.params["ct_clip_range"][1],
                ),
            )
            mask_grid = torchvision.utils.make_grid(masks.unsqueeze(1), 4)
            self.logger.experiment.add_image(f"valid/images", grid, self.global_step)
            self.logger.experiment.add_image(f"valid/masks", mask_grid, self.global_step)

        return {"val_loss": self.loss_f(outputs, masks)}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        tqdm_dict = {"val_loss": val_loss_mean}
        output = {
            "val_loss": val_loss_mean,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }
        return output

    def on_epoch_end(self):
        pass
        # sample_imgs_in_hu = self.dataset.norm.denorm(self.forward(z))
        # grid = torchvision.utils.make_grid(sample_imgs_in_hu, 4, normalize=True)
        # self.logger.experiment.add_image(f"generated_images", grid, self.current_epoch)

    def __prepare_tensor_dataset(self):
        tensor_dataset_path = os.path.join(
            self.metaconf["ws_path"], "tensor_datasets", self.dataset_params.tensor_dataset_name
        )
        # compare configs, if not same, refresh dataset
        current_config_snapshot_exists = H.config_snapshot(
            "dataset_params", self.dataset_params.params, "src/data/aux/.dataset_config_snapshot.json",
        )
        if not current_config_snapshot_exists:
            H.makedirs(tensor_dataset_path)
            _tqdm_kwargs = {"desc": "Preparing TensorDataset", "total": len(self.generic_dataset)}
            for i, sample in tqdm(enumerate(self.generic_dataset), **_tqdm_kwargs):
                f_folder_path = os.path.join(tensor_dataset_path, "0")
                H.makedirs(f_folder_path)
                f_path = os.path.join(tensor_dataset_path, "0", f"nodule_{i}.pt")
                save_nodules = {"nodule": sample["nodule"], "mask": sample["mask"]}
                torch.save(save_nodules, f_path)
        return tensor_dataset_path
