import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import Adam
from torchsummary import summary
import numpy as np

from model_64_64 import create_encoder, create_decoder

class Autoencoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = create_encoder(hparams)
        self.decoder = create_decoder(hparams)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        #return Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = F.mse_loss(output, x)
        self.log('loss', loss, on_step=True, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, "val_input_output")

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, "test_input_vs_reconstruction")

        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss, on_epoch=True, prog_bar=True)


    def save_images(self, x, output, name, n=16):
        """
        Saves a plot of n images from input and output batch
        """
        # make grids and save to logger
        grid_top = vutils.make_grid(x[:n,:,:,:], nrow=n)
        grid_bottom = vutils.make_grid(output[:n,:,:,:], nrow=n)
        grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid, self.current_epoch)

def main(hparams):

    dm = HarbourDataModule(data_dir=hparams.data_root, batch_size=hparams.batch_size)
    dm.setup()

    logger = loggers.TensorBoardLogger(hparams.log_dir, name=f"bs{hparams.batch_size}_nf{hparams.nfe}")

    model = Autoencoder(hparams)

    # print detailed summary with estimated network size
    summary(model, (hparams.nc, hparams.image_size, hparams.image_size), device="cpu")

    trainer = Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model, dm)
    #trainer.test(model)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer, loggers
    from harbour_datamodule import HarbourDataModule


    from PIL import Image
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    import torchvision.transforms as transforms

    parser = ArgumentParser()

    #parser.add_argument("--data_root", type=str, default="data/teejet", help="Train root directory")
    parser.add_argument("--data_root", type=str, default="data/view1_normal/crop0/train", help="Train root directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=64, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=1, help="Number of channels in the training images")
    parser.add_argument("--norm", type=int, default=0, help="Normalize or not")
    parser.add_argument("--nz", type=int, default=16, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=32, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()
    main(args)


'''
        if self.hparams.norm:
            # denormalize images
            denormalization = transforms.Normalize((-self.MEAN / self.STD).tolist(), (1.0 / self.STD).tolist())
            x = [denormalization(i)[2:] for i in x[:n]]
            output = [denormalization(i)[2:] for i in output[:n]]

            # make grids and save to logger
            grid_top = vutils.make_grid(x, nrow=n)
            grid_bottom = vutils.make_grid(output, nrow=n)
        else:
            # make grids and save to logger
            grid_top = vutils.make_grid(x[:n,2:,:,:], nrow=n)
            grid_bottom = vutils.make_grid(output[:n,2:,:,:], nrow=n)
    def prepare_data(self):

        if self.hparams.norm:
            # normalization constants
            if self.hparams.nc == 1:
                self.MEAN = torch.tensor([0.5], dtype=torch.float32)
                self.STD = torch.tensor([0.5], dtype=torch.float32)
            elif self.hparams.nc == 3:
                self.MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
                self.STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


            transform = transforms.Compose(
                [
                    #transforms.Resize(self.hparams.image_size),
                    transforms.Grayscale(),
                    #transforms.CenterCrop(self.hparams.image_size),
                    #transforms.RandomCrop(self.hparams.image_size),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.MEAN.tolist(), self.STD.tolist()),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.Grayscale(),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomVerticalFlip(),
                    #transforms.RandomCrop(self.hparams.image_size),
                    transforms.ToTensor(),
                ]
            )

        dataset = ImageFolder(root=self.hparams.data_root, transform=transform, loader=lambda path: Image.open(path).convert("L"))
        n_sample = len(dataset)
        end_train_idx = int(n_sample * 0.8)
        end_val_idx = int(n_sample * 0.9)
        self.train_dataset = Subset(dataset, range(0, end_train_idx))
        self.val_dataset = Subset(dataset, range(end_train_idx + 1, end_val_idx))
        self.test_dataset = Subset(dataset, range(end_val_idx + 1, n_sample))
'''
