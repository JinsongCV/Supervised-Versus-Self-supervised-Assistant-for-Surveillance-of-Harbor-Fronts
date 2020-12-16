import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os
from glob import glob
import cv2

'''
class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_files = [y for x in os.walk(image_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
        if not len(self.image_files)>0:
            print("did not find any files")

    def load_sample(self, image_path):
        img = cv2.imread(image_path, -1)[:,:,0] # use only the first channel
        return img

    def __getitem__(self, idx):
        img = self.load_sample(self.image_files[idx])
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        img = torch.unsqueeze(img, 0) # (HxW -> C)
        return img

    def __len__(self):
        return len(self.image_files)

class DatasetFromList(torch.utils.data.Dataset):
    def __init__(self, image_list):

        with open(image_list) as f:
            self.image_files = f.read().splitlines()
        if not len(self.image_files)>0:
            print("did not find any files")

    def load_sample(self, image_path):
        img = cv2.imread(image_path, -1)#[:,:,0]
        #img = cv2.resize(img, (64,192), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx):
        img = self.load_sample(self.image_files[idx])
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        img = torch.unsqueeze(img, 0)
        return img

    def __len__(self):
        return len(self.image_files)
'''
class HarbourDataModule(pl.LightningDataModule):
    def __init__(self, image_dir=None, image_lists=None, batch_size=64):
        super().__init__()
        self.image_dir = image_dir
        self.image_lists = image_lists
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.image_dir is not None:
                self.data = DatasetFromFolder(self.image_dir)
            elif self.image_lists is not None:
                self.data_train = DatasetFromList(self.image_lists[0])
                self.data_val = DatasetFromList(self.image_lists[1])
            else:
                print("no input provided!")

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':

    dm = HarbourDataModule(data_dir='data/view1_normal/crop0/train',
                           batch_size=16)

    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
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
