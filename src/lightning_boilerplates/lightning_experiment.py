import logging
import os
from argparse import Namespace

import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import TypeVar

import torch
import torchvision as tv
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from pytorch_lightning.core import LightningModule

import data.transforms as T
import utils.helpers as H
from data.lidc import LIDCNodulesDataset
from models import BaseVAE


log = logging.getLogger("lightning_boilerplates.vae")
Tensor = TypeVar("torch.tensor")


class VAEExperiment(LightningModule):
    def __init__(self, vae_model: BaseVAE, config: Namespace) -> None:
        super().__init__()

        self.model = vae_model
        self.metaconf = config.metaconf
        self.hparams = Namespace(**config.hyperparams)
        self.dataset_params = Namespace(**config.dataloaders["train"])

        self.curr_device = None

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch[0]["nodule"], batch[0]["texture"]
        # real_img = real_img[:, :, real_img.size(2) // 2, :, :]
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss_dict = self.model.loss_function(
            *results,
            M_N=self.hparams.batch_size / self.num_train_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        output = {
            "real_img": real_img,
            "loss": train_loss_dict["loss"],
            # "progress_bar": train_loss_dict,
            "log": {
                "reconstruction_loss": train_loss_dict["Reconstruction_Loss"],
                "KLD": train_loss_dict["KLD"],
            },
        }
        return output

    def training_step_end(self, output):
        if self.global_step % 20 == 0:
            imgs = output["real_img"]
            grid = self.__make_grid(imgs)
            self.logger.experiment.add_image(f"input_images", grid, self.global_step)
        del output["real_img"]
        return output

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch[0]["nodule"], batch[0]["texture"]
        # real_img = real_img[:, :, real_img.size(2) // 2, :, :]
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss_dict = self.model.loss_function(
            *results,
            M_N=self.hparams.batch_size / self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )
        return val_loss_dict

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        self.__sample_images()
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def on_epoch_end(self):
        self.__log_embeddings()

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        if "lr_2" in dir(self.hparams):
            optimizer2 = torch.optim.Adam(
                getattr(self.model, self.hparams["submodel"]).parameters(), lr=self.hparams.lr_2
            )
            optims.append(optimizer2)

        if "scheduler_gamma" in dir(self.hparams):
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optims[0], gamma=self.hparams.scheduler_gamma
            )
            scheds.append(scheduler)

            # Check if another scheduler is required for the second optimizer
            if "scheduler_gamma_2" in dir(self.hparams):
                scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
                    optims[1], gamma=self.hparams.scheduler_gamma_2
                )
                scheds.append(scheduler2)
            return optims, scheds

        return optims

    def prepare_data(self):
        """Prepare and save dataset as TensorDataset to improve training speed.
        """
        self.generic_dataset = LIDCNodulesDataset(**self.dataset_params.params)
        log.info(f"DATASET SIZE: {len(self.generic_dataset)}")

        self.tensor_dataset_path = self.__prepare_tensor_dataset()
        self.aug_transform = transforms.Compose(
            [T.FlipNodule3D(), T.RotNodule3D()]
        )
        self.dataset = DatasetFolder(
            self.tensor_dataset_path, torch.load, ("pt"), transform=self.__data_transform
        )
        self.dataset.norm = self.generic_dataset.norm

        train_inds, val_inds, test_inds = H.train_val_holdout_split(
            self.dataset, ratios=[0.89, 0.1, 0.01]
        )
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
        self.num_train_imgs = len(self.train_sampler)
        return dl

    def val_dataloader(self):
        self.sample_dl = DataLoader(
            self.dataset,
            sampler=self.val_sampler,
            batch_size=self.hparams.batch_size,
            num_workers=self.metaconf["dl_workers"],
        )
        self.num_val_imgs = len(self.val_sampler)
        return self.sample_dl

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
                f_folder_path = os.path.join(tensor_dataset_path, f"{sample['texture']}")
                H.makedirs(f_folder_path)
                f_path = os.path.join(f_folder_path, f"nodule_{i}.pt")
                save_nodules = {"nodule": sample["nodule"], "texture": sample["texture"]}
                torch.save(save_nodules, f_path)
        return tensor_dataset_path

    def __data_transform(self, input):
        image, label = input["nodule"], input["texture"]
        if label != 5:
            plt.imshow(image[0, 32])
            plt.savefig('123.png')
            plt.close()
            image = self.aug_transform(image)
            plt.imshow(image[0, 32])
            plt.savefig('1234.png')
            plt.close()

        image = image[:, image.size(1) // 2, :, :]
        return {"nodule": image, "texture": label}

    def __make_grid(self, samples):
        imgs_in_hu = self.dataset.norm.denorm(samples)
        grid = tv.utils.make_grid(
            imgs_in_hu,
            nrow=8,
            normalize=True,
            range=(
                self.dataset_params.params["ct_clip_range"][0],
                self.dataset_params.params["ct_clip_range"][1],
            ),
        )
        return grid

    def __sample_images(self):
        # Get sample reconstruction image
        batch = next(iter(self.sample_dl))
        test_input, test_label = batch[0]["nodule"], batch[0]["texture"]
        # test_input = test_input[:, :, test_input.size(2) // 2, :, :]
        test_input, test_label = test_input.to(self.curr_device), test_label.to(self.curr_device)

        recons = self.model.generate(test_input, labels=test_label)
        grid = self.__make_grid(recons)
        self.logger.experiment.add_image(f"reconstructed", grid, self.global_step)

        if "sample" in dir(self.model):
            samples = self.model.sample(test_input.size(0), self.curr_device, labels=test_label)
            grid = self.__make_grid(samples)
            self.logger.experiment.add_image(f"sampled", grid, self.global_step)
            del samples
        del test_input, recons, grid

    def __log_embeddings(self):
        dataset = DatasetFolder(self.tensor_dataset_path, torch.load, ("pt"))
        embeds, labels, imgs = [], [], []
        for sample in DataLoader(dataset, batch_size=64):
            img, label = sample[0]["nodule"], sample[0]["texture"]
            img = img[:, :, img.size(2) // 2, :, :]
            img, label = img.to(self.curr_device), label.to(self.curr_device)

            embeds.append(self.model.embed(img, labels=label))
            labels.append(label)

            min, max = self.dataset_params.params["ct_clip_range"]
            img_in_hu = self.generic_dataset.norm.denorm(img)
            img_in_01 = img_in_hu.add(-min).div(max - min + 1e-5)
            imgs.append(img_in_01)

        embeds = torch.cat(embeds, dim=0)
        labels = torch.cat(labels, dim=0).tolist()
        imgs = torch.cat(imgs, dim=0)
        self.logger.experiment.add_embedding(
            embeds, metadata=labels, label_img=imgs, global_step=self.global_step
        )
