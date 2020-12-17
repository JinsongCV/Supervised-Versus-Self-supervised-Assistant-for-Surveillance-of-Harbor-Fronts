import os
import sys
import cv2
import numpy as np
from glob import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F



def test(cfg, dataset='test', groupings=['normal','abnormal'], show=False, save=False):
    sys.path.append('src/')
    from autoencoder import Autoencoder
    model = Autoencoder(cfg)

    model_dir = os.path.join("trained_models/",cfg['experiment'])
    model.encoder = torch.load(os.path.join(model_dir,"encoder.pt"))
    model.decoder = torch.load(os.path.join(model_dir,"decoder.pt"))
    model.encoder.eval()
    model.decoder.eval()

    experiment_dir = os.path.join("experiments/",cfg['experiment'],dataset)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    sys.path.append('data/')
    from harbour_dataset import HarborDataset

    for group in groupings:
        if dataset == 'train':
            data = HarborDataset(img_dir=os.path.join(cfg['train_folder'],group))
        elif dataset == 'val':
            data = HarborDataset(img_dir=os.path.join(cfg['val_folder'],group))
        elif dataset == 'test':
            data = HarborDataset(img_dir=os.path.join(cfg['test_folder'],group))

        if save or show:
            img_h, img_w = data[0].shape[1:3]
            scale = 2
            # Create a plotter class object
            from plotter import Plotter
            plot = Plotter(img_w*scale*3, img_h*scale*2)


        print("number of frames {}".format(len(data)))
        inputs, recs, files, losses, latent = [], [], [], [], []
        for i, sample in enumerate(data):
            img, path = sample
            z = model.encoder(img.unsqueeze(0))
            rec = model.decoder(z)[0]
            loss = F.mse_loss(rec, img)

            input = img[0,:,:].mul(255).byte().numpy()
            inputs.append(input)
            files.append(path)
            latent.append(z[0].detach().numpy().flatten())
            rec = rec[0,:,:].mul(255).byte().numpy()
            recs.append(rec)
            losses.append(loss.item())

            if save or show:
                vis = np.concatenate((input, rec), axis=0)
                vis = cv2.resize(vis, (vis.shape[1]*scale,vis.shape[0]*scale), interpolation = cv2.INTER_NEAREST)
                plot.plot(loss*100000)
                output = np.concatenate((cv2.merge((vis,vis,vis)), plot.plot_canvas), axis=1)
                output = cv2.putText(output, str(i).zfill(4), (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                #if save:
                    #l = "{:.6f}".format(losses[-1])[-6:]
                    #cv2.imwrite(os.path.join(experiment_dir,"{}_l-{}.png".format(str(i).zfill(5),l)),vis)
                if show:
                    cv2.imshow("test",output)
                    key = cv2.waitKey()
                    if key == 27:
                        break

        np.save(os.path.join(experiment_dir,'{}_inputs.npy'.format(group)), inputs)
        np.save(os.path.join(experiment_dir,'{}_recs.npy'.format(group)), recs)
        np.save(os.path.join(experiment_dir,'{}_files.npy'.format(group)), files)
        np.save(os.path.join(experiment_dir,'{}_losses.npy'.format(group)), losses)
        np.save(os.path.join(experiment_dir,'{}_latent.npy'.format(group)), latent)

if __name__ == "__main__":
    cfg = {
           'experiment': 'normal',
           'train_folder': 'data/train1715/normal/',
           'test_folder': 'data/test500/',
           'val_folder': 'data/val143/normal/',
           'image_size': (64,192),
           'max_epoch': 100,
           'gpus': 1,
           'lr': 0.0002,
           'batch_size': 16,
           'nc': 1,
           'nz': 8,
           'nfe': 32,
           'nfd': 32,
           'device': 'cuda',
          }

    test(cfg)
