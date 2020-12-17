from argparse import ArgumentParser
import os
import sys
import cv2
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F


def train(cfg):
    pl.seed_everything(42)
    logger = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs/',cfg['experiment']))

    sys.path.append('data/')
    from harbour_datamodule import HarbourDataModule
    dm = HarbourDataModule(cfg)
    dm.setup(stage='fit')

    sys.path.append('src/')
    from autoencoder import Autoencoder
    model = Autoencoder(cfg)
    trainer = Trainer(gpus=cfg['gpus'], max_epochs=cfg['max_epoch'], logger=logger, deterministic=True)
    trainer.fit(model, dm)

    model_dir = os.path.join("trained_models/",cfg['experiment'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.encoder, os.path.join(model_dir,"encoder.pt"))
    torch.save(model.decoder, os.path.join(model_dir,"decoder.pt"))


if __name__ == "__main__":


    cfg = {
           'experiment': 'normal',
           'train_folder': 'data/train1715/normal/',
           'test_folder': 'data/test500/normal/',
           'val_folder': 'data/val143/normal/',
           'image_size': (64,192),
           'max_epoch': 50,
           'gpus': 1,
           'lr': 0.0005,
           'batch_size': 16,
           'nc': 1,
           'nz': 8,
           'nfe': 32,
           'nfd': 32,
           'device': 'cuda',
          }

    train(cfg)
