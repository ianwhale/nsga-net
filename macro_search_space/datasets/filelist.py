# filelist.py

import os
import math
import utils as utils
import torch.utils.data as data
from sklearn.utils import shuffle
import datasets.loaders as loaders


class FileListLoader(data.Dataset):
    def __init__(self, ifile, lfile=None, split_train=1.0, split_test=0.0, train=True, 
        transform_train=None, transform_test=None, loader_input='image', loader_label='torch'):

        self.ifile = ifile
        self.lfile = lfile
        self.train = train
        self.split_test = split_test
        self.split_train = split_train
        self.transform_test = transform_test
        self.transform_train = transform_train

        if loader_input == 'image':
            self.loader_input = loaders.loader_image
        if loader_input == 'torch':
            self.loader_input = loaders.loader_torch
        if loader_input == 'numpy':
            self.loader_input = loaders.loader_numpy

        if loader_label == 'image':
            self.loader_label = loaders.loader_image
        if loader_label == 'torch':
            self.loader_label = loaders.loader_torch
        if loader_label == 'numpy':
            self.loader_label = loaders.loader_numpy

        if ifile is not None:
            imagelist = utils.readtextfile(ifile)
            imagelist = [x.rstrip('\n') for x in imagelist]
        else:
            imagelist = []

        if lfile is not None:
            labellist = utils.readtextfile(lfile)
            labellist = [x.rstrip('\n') for x in labellist]
        else:
            labellist = []

        if len(imagelist) == len(labellist):
            shuffle(imagelist, labellist)

        if len(imagelist) > 0 and len(labellist) == 0:
            shuffle(imagelist)

        if len(labellist) > 0 and len(imagelist) == 0:
            shuffle(labellist)

        if (self.split_train < 1.0) & (self.split_train > 0.0):
            if len(imagelist) > 0:
                num = math.floor(self.split_train * len(imagelist))
                self.images_train = imagelist[0:num]
                self.images_test = imagelist[num + 1:len(imagelist)]
            else:
                self.images_test = []
                self.images_train = []

            if len(labellist) > 0:
                num = math.floor(self.split * len(labellist))
                self.labels_train = labellist[0:num]
                self.labels_test = labellist[num + 1:len(labellist)]
            else:
                self.labels_test = []
                self.labels_train = []

        elif self.split_train == 1.0:
            if len(imagelist) > 0:
                self.images_train = imagelist
            else:
                self.images_train = []
            if len(labellist) > 0:
                self.labels_train = labellist
            else:
                self.labels_train = []

        elif self.split_test == 1.0:
            if len(imagelist) > 0:
                self.images_test = imagelist
            else:
                self.images_test = []
            if len(labellist) > 0:
                self.labels_test = labellist
            else:
                self.labels_test = []

    def __len__(self):
        if self.train is True:
            return len(self.images_train)
        if self.train is False:
            return len(self.images_test)

    def __getitem__(self, index):
        if self.train is True:
            if len(self.images_train) > 0:
                path = self.images_train[index]
                image = self.loader_input(path)
            else:
                image = []

            if len(self.labels_train) > 0:
                label = self.labels_train[index]
            else:
                label = []

            if self.transform_train is not None:
                image = self.transform_train(image)

        if self.train is False:
            if len(self.images_test) > 0:
                path = self.images_test[index]
                image = self.loader_input(path)
            else:
                image = []

            if len(self.labels_test) > 0:
                label = self.labels_test[index]
            else:
                label = []

            if self.transform_test is not None:
                image = self.transform_test(image)

            path = os.path.basename(path)
        return image, label, path
