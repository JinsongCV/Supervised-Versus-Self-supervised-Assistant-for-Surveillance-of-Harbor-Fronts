import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import Adam
from torchsummary import summary
import numpy as np

from model import create_encoder, create_decoder

class Autoencoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = create_encoder(cfg)
        self.decoder = create_decoder(cfg)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.cfg['lr'])

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)
        self.log('loss', loss, on_step=True, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
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


#if __name__ == "__main__":
#    summary(model, (hparams.nc, hparams.image_size, hparams.image_size), device="cpu")
