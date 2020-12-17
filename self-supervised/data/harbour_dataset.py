import os
import numpy as np
import torch
import torch.utils.data
import argparse

import cv2
import random
import math
from glob import glob

def four_point_transform(tl, tr, br, bl):
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

    rect = np.zeros((4, 2), dtype = "float32")
    rect[0], rect[1], rect[2], rect[3] = tl, tr, br, bl

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    return M, maxWidth, maxHeight

class HarborDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, crop_size=(64,192)):
        self.crop_size = crop_size
        self.image_list = sorted([y for x in os.walk(img_dir) for y in glob(os.path.join(x[0], '*.jpg'))])
        if not len(self.image_list)>0:
            print("did not find any files")

        # points from data/roi.txt
        self.M, self.w_max, self.h_max = four_point_transform(tl=[52,108], tr=[67,108], br=[170,265], bl=[112,270])

    def load_image(self, image_path):
        thermal = cv2.imread(image_path)
        self.image_h, self.image_w, _ = thermal.shape
        thermal = thermal[:,:,0]
        return thermal, image_path

    def __getitem__(self, idx):
        img, path = self.load_image(self.image_list[idx])
        # warped roi to square
        img = cv2.warpPerspective(img, self.M, (self.w_max, self.h_max))
        img = cv2.resize(img, self.crop_size, interpolation=cv2.INTER_LINEAR)
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        img = torch.unsqueeze(img, 0) # (HxW -> CxHxW)
        return img, path

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    #dataset = HarborDataset(img_dir='data/test500/')
    dataset = HarborDataset(img_dir='data/test500/normal409/')

    output_dir = 'data/output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(len(dataset))
    for i, crop in enumerate(dataset):
        crop = crop[0,:,:].mul(255).byte().numpy()
        cv2.imwrite(os.path.join(output_dir,'{}.jpg'.format(str(i).zfill(5))),crop)
