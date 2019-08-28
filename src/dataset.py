import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb


class Dataset(torch.utils.data.Dataset):
    def __init__(self, lr_flist, hr_flist, sigma, scale, hr_size, augment=True):
        super().__init__()
        self.augment = augment
        self.lr_data = self.load_flist(lr_flist)
        self.hr_data = self.load_flist(hr_flist)

        self.hr_size = hr_size
        self.sigma = sigma
        self.scale = scale

    def __len__(self):
        return len(self.hr_data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.hr_data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.hr_data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.hr_size
        scale = self.scale

        # load hr image
        hr_img = imread(self.hr_data[index])


        # load lr image
        if len(self.lr_data) != 0:
            lr_img = imread(self.lr_data[index])
        # create lr image
        else:
            lr_img = hr_img
            if size == 0:
                # resize without cropping
                imgh, imgw = lr_img.shape[0:2]
                lr_img = scipy.misc.imresize(lr_img, [imgh // scale, imgw // scale])


        # resize/crop if needed
        if size != 0:
            hr_img = self.resize(hr_img, size, size)
            lr_img = self.resize(lr_img, size // scale, size // scale)

        # load edge
        hr_edge = self.load_edge(hr_img, index)
        lr_edge = self.load_edge(lr_img, index)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            hr_img = hr_img[:, ::-1, ...]
            lr_img = lr_img[:, ::-1, ...]
            hr_edge = hr_edge[:, ::-1, ...]
            lr_edge = lr_edge[:, ::-1, ...]

        return self.to_tensor(lr_img), self.to_tensor(hr_img), self.to_tensor(lr_edge), self.to_tensor(hr_edge)

    def load_edge(self, img, index):
        return canny(rgb2gray(img), sigma=self.sigma).astype(np.float)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width):
        imgh, imgw = img.shape[0:2]

        if imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
