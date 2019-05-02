# folderlist.py

import os
import math
import pickle
import os.path
import utils as utils
import torch.utils.data as data
from sklearn.utils import shuffle
import datasets.loaders as loaders

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(classlist):
    filename = '/tmp/folderlist.pkl'
    if utils.file_exists(filename):
        print("loading from cache")
        pickle_load = pickle.load(open(filename, "rb"))
        images = pickle_load["images"]
        labels = pickle_load["labels"]
    else:
        print("cache not found, generating cache, this will take a while")
        images = []
        labels = []
        classes = utils.readtextfile(classlist)
        classes = [x.rstrip('\n') for x in classes]
        classes.sort()

        for index in range(len(classes)):
            for fname in os.listdir(classes[index]):
                if is_image_file(fname):
                    fname = os.path.join(classes[index], fname)
                    images.append(fname)
                    labels.append(index)

        pickle_save = {"images": images, "labels": labels}
        pickle.dump(pickle_save, open(filename, "wb"))
    return images, labels


class FolderListLoader(data.Dataset):
    def __init__(self, ifile, split_train=1.0, split_test=0.0, train=True,
                 transform_train=None, transform_test=None,
                 loader_input='image', loader_label='torch', prefetch=False):

        self.train = train
        self.prefetch = prefetch

        imagelist, labellist = make_dataset(ifile)
        if len(imagelist) == 0:
            raise(RuntimeError("No images found"))
        if len(labellist) == 0:
            raise(RuntimeError("No labels found"))

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

        self.transform_test = transform_test
        self.transform_train = transform_train

        self.split_test = split_test
        self.split_train = split_train

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
                num = math.floor(self.split_train * len(labellist))
                self.labels_train = labellist[0:num]
                self.labels_test = labellist[num + 1:len(labellist)]
            else:
                self.labels_test = []
                self.labels_train = []

        if (self.split_test < 1.0) & (self.split_test > 0.0):
            if len(imagelist) > 0:
                num = math.floor(self.split_test * len(imagelist))
                self.images_train = imagelist[0:num]
                self.images_test = imagelist[num + 1:len(imagelist)]
            else:
                self.images_test = []
                self.images_train = []
            if len(labellist) > 0:
                num = math.floor(self.split_test * len(labellist))
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

        if self.prefetch is True:
            print("Prefetching is on, loading all data to memory")
            for i in range(len(self.images_test)):
                path = self.images_test[i]
                self.images_test[i] = self.loader_input(path)
            for i in range(len(self.images_train)):
                path = self.images_train[i]
                self.images_train[i] = self.loader_input(path)

    def __len__(self):
        if self.train is True:
            return len(self.images_train)
        if self.train is False:
            return len(self.images_test)

    def __getitem__(self, index):
        if self.train is True:
            if len(self.images_train) > 0:
                if self.prefetch is True:
                    image = self.images_train[index]
                else:
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
                if self.prefetch is True:
                    image = self.images_test[index]
                else:
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
            else:
                image = []

        return image, label
