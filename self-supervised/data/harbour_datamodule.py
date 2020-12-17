import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os
from glob import glob
import cv2

from harbour_dataset import HarborDataset

class HarbourDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.data_train = HarborDataset(img_dir=self.cfg['train_folder'])
            self.data_val = HarborDataset(img_dir=self.cfg['val_folder'])
        else:
            self.data_test = HarborDataset(img_dir=self.cfg['test_folder'])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.cfg['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.cfg['batch_size'], shuffle=False)


if __name__ == '__main__':

    dm = HarbourDataModule(data_dir='data/train1715/normal/',
                           batch_size=16)
    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "data/output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.val_dataloader()):
        imgs = batch
        for img in imgs:
            #img = img.mul(255).permute(1, 2, 0).byte().numpy()
            img = img.mul(255).byte().numpy()
            output_dir = os.path.join(output_root,str(batch_id).zfill(6))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = "id-{}.png".format(str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),img)
            sample_idx = sample_idx + 1
