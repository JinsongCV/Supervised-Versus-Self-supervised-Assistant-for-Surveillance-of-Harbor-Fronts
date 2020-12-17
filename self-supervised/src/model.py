import torch
import torch.nn as nn

def create_encoder(cfg, kernel_size=4):
    return nn.Sequential(
        # input (nc) x 64 x 64 (minimum)
        nn.Conv2d(cfg['nc'], cfg['nfe'], kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nfe']),
        nn.LeakyReLU(True),
        # input (nfe) x 32 x 32
        nn.Conv2d(cfg['nfe'], cfg['nfe'] * 2, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nfe'] * 2),
        nn.LeakyReLU(True),
        # input (nfe*2) x 16 x 16
        nn.Conv2d(cfg['nfe'] * 2, cfg['nfe'] * 4, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nfe'] * 4),
        nn.LeakyReLU(True),
        # input (nfe*4) x 8 x 8
        nn.Conv2d(cfg['nfe'] * 4, cfg['nfe'] * 8, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nfe'] * 8),
        nn.LeakyReLU(True),
        # input (nfe*8) x 4 x 4
        nn.Conv2d(cfg['nfe'] * 8, cfg['nz'], kernel_size, 1, 0, bias=False),
        nn.BatchNorm2d(cfg['nz']),
        nn.LeakyReLU(True)
        # output (nz) x 1 x 1
    )

def create_decoder(cfg, kernel_size=4):
    return nn.Sequential(
        # input (nz) x 1 x 1
        nn.ConvTranspose2d(cfg['nz'], cfg['nfd'] * 8, kernel_size, 1, 0, bias=False),
        nn.BatchNorm2d(cfg['nfd'] * 8),
        nn.ReLU(True),
        # input (nfd*8) x 4 x 4
        nn.ConvTranspose2d(cfg['nfd'] * 8, cfg['nfd'] * 4, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nfd'] * 4),
        nn.ReLU(True),
        # input (nfd*4) x 8 x 8
        nn.ConvTranspose2d(cfg['nfd'] * 4, cfg['nfd'] * 2, kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nfd'] * 2),
        nn.ReLU(True),
        # input (nfd*2) x 16 x 16
        nn.ConvTranspose2d(cfg['nfd'] * 2, cfg['nfd'], kernel_size, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nfd']),
        nn.ReLU(True),
        # input (nfd) x 32 x 32
        nn.ConvTranspose2d(cfg['nfd'], cfg['nc'], kernel_size, 2, 1, bias=False),
        nn.Sigmoid()
        # output (nc) x 64 x 64 (minimum)
    )
