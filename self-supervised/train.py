from argparse import ArgumentParser
import os
import cv2
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F

from autoencoder import Autoencoder
from harbour_datamodule import HarbourDataModule

def train(hparams, dm):
    model = Autoencoder(hparams)
    trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model, dm)
    torch.save(model.encoder, "trained_models/encoder.pt")
    torch.save(model.decoder, "trained_models/decoder.pt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_list", type=str, default="output/train_total.txt", help="list of training images")
    parser.add_argument("--val_list", type=str, default="output/val_all.txt", help="list of validation images")
    parser.add_argument("--num_workers", type=int, default=12, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_width", type=int, default=64, help="Width of images")
    parser.add_argument("--image_height", type=int, default=192, help="Height of images")
    parser.add_argument("--max_epochs", type=int, default=300, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=1, help="Number of channels in the training images")
    parser.add_argument("--norm", type=int, default=0, help="Normalize or not")
    parser.add_argument("--nz", type=int, default=8, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=32, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()

    dm = HarbourDataModule(image_lists=[args.val_list,args.train_list], batch_size=args.batch_size)
    dm.setup()

    train(args, dm)
