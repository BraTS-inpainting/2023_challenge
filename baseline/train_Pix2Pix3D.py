# @author Robert Graf; deep-spine.de 2023
from pathlib import Path
import numpy

import torch
import torchvision
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure

from dataset3D import Dataset_Training
from models.pix2pix_model import Discriminator
from models.unet import UNet
from baseline_utils import LambdaLR, denormalize
import argparse  # for nice command line argument parsing

import pytorch_lightning as pl


class Pix2Pix3D(pl.LightningModule):
    def __init__(
        self,
        epochs,
        in_channels=2,
        out_channels=1,
        decay_epoch=-1,
        lambda_paired=10,
        lambda_ssim=1,
        lambda_GAN=1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.logger: TensorBoardLogger
        ### Model ###
        self.model = UNet(input_nc=in_channels, output_nc=out_channels, net_G_channel=32, net_G_depth=6, net_G_drop_out=0)
        self.discriminator = Discriminator(in_channels + out_channels, depth=3, channels=64)
        self.criterion = nn.L1Loss()
        #### Losses ####
        self.criterion_paired = torch.nn.L1Loss()
        # Using LSGAN variants hardcoded
        self.criterion_GAN = torch.nn.MSELoss()

        self.counter = 0
        self.lr = 1e-5
        self.epochs = epochs
        self.decay_epoch = decay_epoch
        self.lambda_paired = lambda_paired
        self.lambda_ssim = lambda_ssim
        self.lambda_GAN = lambda_GAN
        self.automatic_optimization = False

        self.losses = []

    def forward(self, voided_image: Tensor, mask: Tensor) -> Tensor:
        if self.in_channels == 2:
            x = torch.cat([voided_image, mask], 1)
        else:
            x = voided_image
        return self.model(x) * mask + voided_image * (~mask)

    def configure_optimizers(self):
        lr = self.lr
        optimizer_G = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        if self.decay_epoch == -1:
            self.decay_epoch = self.epochs // 2

        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.epochs, 0, self.decay_epoch).step)
        lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(self.epochs, 0, self.decay_epoch).step)

        return [optimizer_G, optimizer_D], [lr_scheduler_G, lr_scheduler_D]

    def training_step(self, train_batch, batch_idx) -> Tensor:
        #### Optimizer ####
        (opt_a, opt_b) = self.optimizers()  # type: ignore
        #### Get batch ####
        voided_image = train_batch["voided_healthy_image"]
        gt_image = train_batch["gt_image"]
        mask = train_batch["healthy_mask"]

        # Assertions
        if len(voided_image.shape) != 5 or len(gt_image.shape) != 5 or len(mask.shape) != 5:
            raise UserWarning(
                f"Input data needs to have 5 dimensions: N, C, X, Y, Z. Current shapes: {voided_image.shape}, {gt_image.shape}, {mask.shape} "
            )

        # Compute forward loss.
        loss_gan = self.training_step_G(voided_image, gt_image, mask)
        self.manual_backward(loss_gan)
        opt_a.step()
        # Compute discriminator loss.
        loss_d = self.training_step_D(voided_image, gt_image, mask)
        self.manual_backward(loss_d)
        opt_b.step()
        return loss_gan

    def training_step_G(self, voided_image: Tensor, gt_image: Tensor, mask: Tensor):
        fake_gt_image: Tensor = self.forward(voided_image, mask)

        if self.in_channels == 2:
            voided_image = torch.cat([voided_image, mask], 1)

        fake = torch.cat([fake_gt_image, voided_image], dim=1)
        loss_G_GAN = 0
        if self.lambda_GAN > 0.0:
            pred_fake: Tensor = self.discriminator(fake)
            real_label = torch.ones((pred_fake.shape[0], 1), device=self.device)
            loss_G_GAN = self.criterion_GAN(pred_fake, real_label) * self.lambda_GAN
            self.log(f"train/loss_GAN", loss_G_GAN.detach().cpu())
        loss_paired = 0
        loss_ssim = 0
        # compute MSE only on
        gt_image_values = gt_image[mask]
        fake_gt_image_values = fake_gt_image[mask]
        if len(gt_image_values) == 0:  # if there is no infill region in this cuboid just return loss 0.
            loss_paired = self.criterion(gt_image, fake_gt_image)  # TODO: find better solution!
        else:  # otherwise evaluate loss properly
            loss_paired = self.criterion_paired(gt_image_values, fake_gt_image_values)

        self.log(f"train/loss_paired", loss_paired.detach().cpu())
        if self.lambda_ssim > 0.0:
            loss_ssim = self.lambda_ssim * (1 - structural_similarity_index_measure(gt_image + 1, fake_gt_image + 1, data_range=2.0))  # type: ignore
            self.log(f"train/loss_ssim", loss_ssim.detach().cpu())
        loss_paired = self.lambda_paired * (loss_ssim + loss_paired)

        self.fake_B_buffer = fake.detach()

        loss_G = loss_G_GAN + loss_paired
        self.log(f"train/All", loss_G.detach().cpu())
        # print avg loss into progress_bar
        self.losses.append(loss_G.detach().cpu())
        self.losses.pop(0) if len(self.losses) > 250 else None
        self.log("loss", numpy.mean(self.losses), prog_bar=True)  # type: ignore #TODO: why is here numpy not torch?

        return loss_G

    def training_step_D(self, voided_image, gt_image, mask) -> Tensor:
        if self.in_channels == 2:
            voided_image = torch.cat([voided_image, mask], 1)

        # This code will only update the discriminator because of the settings of the second optimizer
        # Fake loss, will be fake_B if unpaired and fake_B||real_A if paired
        fake = self.fake_B_buffer
        pred_fake = self.discriminator(fake)
        fake_label = torch.zeros((pred_fake.shape[0], 1), device=self.device)
        loss_D_fake = self.criterion_GAN(pred_fake, fake_label).mean()  # is mean really necessary?
        # Real loss
        real = torch.cat([gt_image, voided_image], dim=1)
        pred_real = self.discriminator(real)
        real_label = torch.ones((pred_real.shape[0], 1), device=self.device)
        loss_D_real = self.criterion_GAN(pred_real, real_label).mean()
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.log(f"train/Discriminator", loss_D.detach().cpu())
        return loss_D

    def validation_step(self, batch, batch_idx):
        voided_image = batch["voided_healthy_image"]
        gt_image = batch["gt_image"]
        mask = batch["healthy_mask"]

        # Assertions
        if len(voided_image.shape) != 5 or len(gt_image.shape) != 5 or len(mask.shape) != 5:
            raise UserWarning(
                f"Input data needs to have 5 dimensions: N, C, X, Y, Z. Current shapes: {voided_image.shape}, {gt_image.shape}, {mask.shape} "
            )

        fake_B = self.forward(voided_image, mask)

        out = [gt_image, fake_B, batch["voided_healthy_image"]]
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
    # Checkpoint path
    parser.add_argument("-cp", "--ckpt-path", dest="ckpt_path", type=str, default=None, help="Checkpoint path")

    #### Set Parameters / Initialize ###
    args = parser.parse_args()  # Get commandline arguments
    print(args)
    name = "Pix2Pix3D"
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

    print(f"Seed: {seed}")
    # Randomness (just seed everything. Will probably only be relevant for the train-val split though...)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # Get dataset and split it
    if not dataset_path.exists():
        raise UserWarning(f'Dataset path "{dataset_path}" does not exist!!')
    dataset = Dataset_Training(dataset_path, crop_shape=crop_shape, center_on_mask=True)
    train_set, validation_set = torch.utils.data.random_split(dataset, [train_p, val_p])
    #Note: if the above line (random_split) results in "ValueError: Sum of input lengths does not equal the length of the input dataset!"
    # for the original version of the notebook/code, you are probably using torch version below 1.13 which is not supporting fractions
    # see https://github.com/BraTS-inpainting/2023_challenge/issues/1 

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=16)
    # TODO: tune prefetch_factor and pin_memory

    ### Model and Training ###
    lighting_module = Pix2Pix3D(epochs)
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    # Define Last and best Checkpoints to be saved.
    checkpoint_callback = ModelCheckpoint(
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
        callbacks=[checkpoint_callback],  # You may add here additional call back that saves the best model
        # limit_train_batches=150
        # detect_anomaly=True,
        strategy=("ddp_find_unused_parameters_true" if len(gpus) > 1 else "auto"),  # for distributed compatibility
    )

    # Fit/train model
    if ckpt_path != None:  # try to continue training
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise UserWarning(f'Checkpoint path "{ckpt_path}" does not exist!!')
        print(f"Try to resume from {ckpt_path}")
        trainer.fit(lighting_module, train_loader, validation_loader, ckpt_path=ckpt_path)
    else:  # start training anew
        trainer.fit(lighting_module, train_loader, validation_loader)
