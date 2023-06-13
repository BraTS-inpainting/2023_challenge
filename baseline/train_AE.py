# @author Robert Graf; deep-spine.de 2023
from pathlib import Path
import numpy

import torch
import torchvision
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from dataset3D import Dataset_Training
from models.pix2pix_model import Generator
from baseline_utils import denormalize
import argparse  # for nice command line argument parsing


import pytorch_lightning as pl


class AutoEncoder(pl.LightningModule):
    def __init__(self, in_channels=2, out_channels=1, net_G_channel=32, net_G_depth=4, net_G_drop_out=0.0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.logger: TensorBoardLogger
        ### Model ###
        self.model = Generator(
            in_channels,
            out_channels,
            net_G_channel=net_G_channel,
            net_G_depth=net_G_depth,
            net_G_drop_out=net_G_drop_out,
        )
        self.criterion = nn.L1Loss()
        #### Losses ####
        self.criterion_paired = torch.nn.L1Loss()

        self.counter = 0
        self.lr = 1e-5
        self.losses = []

    def forward(self, voided_image: Tensor, mask: Tensor) -> Tensor:
        if self.in_channels == 2:
            x = torch.cat([voided_image, mask], 1)
        else:
            x = voided_image
        return self.model(x) * mask + voided_image * (~mask)

    def configure_optimizers(self):
        # This is the used Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=max(self.lr, 1e-22))
        return optimizer

    def training_step(self, train_batch, batch_idx) -> Tensor:
        voided_image = train_batch["voided_healthy_image"]
        gt_image = train_batch["gt_image"]
        mask = train_batch["healthy_mask"]

        outputs = self.forward(voided_image, mask)

        # only evaluate metrics on the infill regions, not the complete cuboid
        gt_image_values = gt_image[mask]
        outputs_values = outputs[mask]

        if len(gt_image_values) == 0:  # if there is no infill region in this cuboid just retun loss 0.
            train_loss = self.criterion(gt_image, outputs)  # TODO: find better solution!
        else:  # otherwise evaluate loss properly
            train_loss = self.criterion(gt_image_values, outputs_values)

        self.log(f"train/loss_paired", train_loss.detach().cpu())

        # print avg loss into progress_bar
        self.losses.append(train_loss.detach().cpu())
        self.losses.pop(0) if len(self.losses) > 250 else None
        self.log("loss", numpy.mean(self.losses), prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        voided_image = batch["voided_healthy_image"]
        gt_image = batch["gt_image"]
        mask = batch["healthy_mask"]
        outputs = self.forward(voided_image, mask)

        out = [gt_image, outputs, batch["voided_healthy_image"]]
        out = [denormalize(i) for i in out]
        # Manipulate so we get a top down view.
        out = [i[:, 0].swapaxes(0, 1).swapaxes(0, -1) for i in out]

        grid = torch.cat(out, dim=-1).cpu()
        grid = torchvision.utils.make_grid(grid, nrow=10)

        self.logger.experiment.add_image("A2B", grid, self.counter)

        self.counter += 1
        # TODO You may compute a validation loss for early stopping or selecting best model. This must be done on real validation data!


if __name__ == "__main__":
    ### Setup argument parser ###
    parser = argparse.ArgumentParser(description="Description of the argument parser.")
    # Training Epochs
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, help="Epochs for trainings", default=500)
    # Batch Size
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, help="Batch size per iteration", default=500)
    # Dataset
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset_path",
        type=str,
        default="../ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training",
        help="BraTS2023 training datatset path",
    )
    # Accelerator
    parser.add_argument("-acc", "--accelerator", dest="accelerator", type=str, default="gpu", help="Pytroch Lightning accelerator")
    # GPUs
    parser.add_argument(
        "-g",
        "--gpus",
        dest="gpus",
        type=int,
        nargs="+",
        default=[0],
        help="comma separated list of cuda device (e.g. GPUs) to be used",
    )
    # Seed
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=2023, help="Seed for dataset splitting")
    # Train-Validation split percentage
    parser.add_argument("-ds", "--split", dest="split", type=float, nargs="+", default=(0.8, 0.2), help="Dataset split")
    # Train-Validation split percentage
    parser.add_argument("-cs", "--crop-shape", dest="crop_shape", type=int, nargs="+", default=[128, 128, 96], help="Crop shape")
    # Accelerator
    parser.add_argument("-cp", "--ckpt-path", dest="ckpt_path", type=str, default=None, help="Checkpoint path")

    #### Set Parameters / Initialize ###
    args = parser.parse_args()  # Get commandline arguments
    print(args)
    name = "AutoEncoder"
    epochs = args.epochs
    batch_size = args.batch_size
    dataset_path = Path(args.dataset_path)
    accelerator = args.accelerator
    gpus = args.gpus  # Or list of GPU-Ids; 0 is CPU
    print(f"Running on {accelerator}:")
    for gpu in gpus:
        print(f"\t[{gpu}]: {torch.cuda.get_device_name(gpu)}")
    seed = args.seed
    train_p, val_p = args.split
    crop_shape = args.crop_shape
    if args.ckpt_path == "None":
        ckpt_path = None
    else:
        ckpt_path = args.ckpt_path
    auto_lr_find = True

    print(seed)
    # Randomness (just seed everything. Will probably only be relevant for the train-val split though...)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # Get dataset and split it
    if not dataset_path.exists():
        raise UserWarning(f'Dataset path "{dataset_path}" does not exist!!')
    dataset = Dataset_Training(dataset_path, crop_shape=crop_shape, center_on_mask=True)
    train_set, validation_set = torch.utils.data.random_split(dataset, [train_p, val_p])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=16)

    ### Model and Training ###
    lighting_module = AutoEncoder()
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    # Define Last and best Checkpoints to be saved.
    mc_last = ModelCheckpoint(
        filename="{epoch}-{step}_latest",
        monitor="step",
        mode="max",
        every_n_train_steps=min(100, len(train_loader)),
        save_top_k=3,
    )

    # Create TensorBoardLogger
    logger = TensorBoardLogger("lightning_logs", name=name, default_hp_metric=False)
    print("######################################")
    print("experiment_name:", name)
    print("######################################")

    # Setup Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=gpus,
        num_nodes=1,
        # limit_train_batches=limit_train_batches,  # Train only x % (if float) or train only on x batches (if int)
        limit_val_batches=3,
        max_epochs=epochs,  # Stopping epoch
        logger=logger,
        callbacks=[mc_last],  # You may add here additional call back that saves the best model
        # limit_train_batches=150
        # detect_anomaly=True,
        strategy=("ddp_find_unused_parameters_true" if len(gpus) > 1 else "auto"),  # for distributed compatibility
    )

    # Setup/apply automatic lr finder
    if auto_lr_find:
        from pytorch_lightning.tuner.tuning import Tuner

        Tuner(trainer).lr_find(lighting_module, train_loader, validation_loader)
        try:
            next(Path().glob(".lr_find*")).unlink()
        except StopIteration:
            pass

    # Fit/train model
    if ckpt_path != None:  # try to continue training
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise UserWarning(f'Checkpoint path "{ckpt_path}" does not exist!!')
        print(f"Try to resume from {ckpt_path}")
        trainer.fit(lighting_module, train_loader, validation_loader, ckpt_path=ckpt_path)
    else:  # start training anew
        trainer.fit(lighting_module, train_loader, validation_loader)
