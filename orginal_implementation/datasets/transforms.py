# transforms.py

from __future__ import division

import math
import types
import torch
import random
import numbers
import numpy as np
from PIL import Image, ImageOps


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)

        return input


class ToTensor(object):
    """Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range
    [0.0, 1.0]."""

    def __call__(self, input):
        for key in input.keys():
            value = input[key]
            if isinstance(value, np.ndarray):
                # handle numpy array
                input[key] = torch.from_numpy(value)
            else:
                # handle PIL Image
                tmp = torch.ByteTensor(torch.ByteStorage.from_buffer(value.tobytes()))
                value = tmp.view(value.size[1], value.size[0], len(value.mode))
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                value = value.transpose(0, 1).transpose(0, 2).contiguous()
                input[key] = value.float().div(255)     
        return input


class ToPILImage(object):
    """Converts a torch.*Tensor of range [0, 1] and shape C x H x W
    or numpy ndarray of dtype=uint8, range[0, 255] and shape H x W x C
    to a PIL.Image of range [0, 255]
    """
    def __call__(self, input):
        if isinstance(input['img'], np.ndarray):
            # handle numpy array
            input['img'] = Image.fromarray(input['img'])
        else:
            npimg = input['img'].mul(255).byte().numpy()
            npimg = np.transpose(npimg, (1, 2, 0))
            input['img'] = Image.fromarray(npimg)
        return input


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        # TODO: make efficient
        for t, m, s in zip(input['img'], self.mean, self.std):
            t.sub_(m).div_(s)
        return input


class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input):
        w, h = input['img'].size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return input
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            input['img'] = input['img'].resize((ow, oh), self.interpolation)
            return input
        else:
            oh = self.size
            ow = int(self.size * w / h)
            input['img'] = input['img'].resize((ow, oh), self.interpolation)
            return input


class CenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, input):
        w, h = input['img'].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        input['img'] = input['img'].crop((x1, y1, x1 + tw, y1 + th))
        return input


class Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""
    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number)
        self.padding = padding
        self.fill = fill

    def __call__(self, input):
        input['img'] = ImageOps.expand(input['img'], border=self.padding, fill=self.fill)
        return input


class Lambda(object):
    """Applies a lambda as a transform."""
    def __init__(self, lambd):
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def __call__(self, input):
        input['img'] = self.lambd(input['img'])
        return input


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, input):
        if self.padding > 0:
            input['img'] = ImageOps.expand(input['img'], border=self.padding, fill=0)

        w, h = input['img'].size
        th, tw = self.size
        if w == tw and h == th:
            return input

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        input['img'] = input['img'].crop((x1, y1, x1 + tw, y1 + th))
        return input


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, input):
        if random.random() < 0.5:
            input['img'] = input['img'].transpose(Image.FLIP_LEFT_RIGHT)
            input['tgt'] = input['tgt'].transpose(Image.FLIP_LEFT_RIGHT)
            input['loc'][0] = input['loc'][0] - math.ceil(input['img'].size[0] / 2)
        return input


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input):
        for attempt in range(10):
            area = input['img'].size[0] * input['img'].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= input['img'].size[0] and h <= input['img'].size[1]:
                x1 = random.randint(0, input['img'].size[0] - w)
                y1 = random.randint(0, input['img'].size[1] - h)

                input['img'] = input['img'].crop((x1, y1, x1 + w, y1 + h))
                assert(input['img'].size == (w, h))
                input['img'] = input['img'].resize((self.size, self.size), self.interpolation)
                return input

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(input))


class NormalizeLandmarks(object):
    """ max-min normalization of landmarks to range [-1,1]"""
    def __init__(self, xsize, ysize):
        self.xsize = xsize
        self.ysize = ysize

    def __call__(self, input):
        valid_points = [v for v in input['loc'] if v[0] != 0 and v[1] != 0]
        mean = np.mean(valid_points,axis = 0)
        for i in range(input['loc'].shape[0]):
            input['loc'][i][0] = -1 + (input['loc'][i][0] * 2. )/(inputx_res)
            input['loc'][i][1] = -1 + (input['loc'][i][1] * 2. )/(inputy_res)
        
        return input


class AffineCrop(object):
    def __init__(self,nlandmark,ix,iy,ox,oy,rangle=0,rscale=0,rtrans=0,gauss=1):
        self.rangle=rangle
        self.rscale=rscale
        self.rtrans=rtrans
        self.nlandmark=nlandmark
        self.ix = ix
        self.iy = iy
        self.ox = ox
        self.oy = oy
        self.utils = utils
        self.gauss = gauss

    def __call__(self, input):

        angle = self.rangle*(2*torch.rand(1)[0] - 1)
        grad_angle  = angle * math.pi / 180
        scale = 1+self.rscale*(2*torch.rand(1)[0] - 1)
        transx = self.rtrans*(2*torch.rand(1)[0] - 1)
        transy = self.rtrans*(2*torch.rand(1)[0] - 1)
        
        img = input['img']
        size = img.size
        h, w = size[0], size[1]
        centerX, centerY = int(w/2), int(h/2)

        # perform rotation
        img = img.rotate(angle, Image.BICUBIC)
        # perform translation
        img = img.transform(img.size, Image.AFFINE, (1, 0, transx, 0, 1, transy))
        # perform scaling
        img = img.resize((int(math.ceil(scale*h)) , int(math.ceil(scale*w))) , Image.ANTIALIAS)

        w, h = img.size
        x1 = int(round((w - self.ix) / 2.))
        y1 = int(round((h - self.ix) / 2.))
        input['img'] = img.crop((x1, y1, x1 + self.ix, y1 + self.iy))
        
        if (np.sum(input['loc']) != 0):
            
            occ = input['occ']
            loc = input['loc']
            newloc = np.ones((3,loc.shape[1]+1))
            newloc[0:2,0:loc.shape[1]] = loc
            newloc[0,loc.shape[1]] = centerY
            newloc[1,loc.shape[1]] = centerX
            
            trans_matrix = np.array([[1,0,-1*transx],[0,1,-1*transy],[0,0,1]])
            scale_matrix = np.array([[scale,0,0],[0,scale,0],[0,0,1]])
            angle_matrix = np.array([[math.cos(grad_angle),math.sin(grad_angle),0],[-math.sin(grad_angle),math.cos(grad_angle),0],[0,0,1]])

            # perform rotation
            newloc[0,:] = newloc[0,:] - centerY
            newloc[1,:] = newloc[1,:] - centerX
            newloc = np.dot(angle_matrix, newloc)
            newloc[0,:] = newloc[0,:] + centerY
            newloc[1,:] = newloc[1,:] + centerX
            # perform translation
            newloc = np.dot(trans_matrix, newloc)
            # perform scaling
            newloc = np.dot(scale_matrix, newloc)

            newloc[0,:] = newloc[0,:] - y1
            newloc[1,:] = newloc[1,:] - x1
            input['loc'] = newloc[0:2,:]
            
            for i in range(input['loc'].shape[1]):
                if ~((input['loc'][0, i] == np.nan) & (input['loc'][1,i] == np.nan)):
                    if ((input['loc'][0, i] < 0) | (input['loc'][0, i] > self.iy) | (input['loc'][1, i] < 0) | (input['loc'][1, i] > self.ix)):
                        input['loc'][:, i] = np.nan
                        input['occ'][i] = 0

        # generate heatmaps
        input['tgt'] = np.zeros((self.nlandmark+1, self.ox, self.oy))
        for i in range(self.nlandmark):
            if  (not np.isnan(input['loc'][:,i][0]) and not np.isnan(input['loc'][:,i][1])):
                tmp = self.utils.gaussian(np.array([self.ix,self.iy]),input['loc'][:,i],self.gauss)
                scaled_tmp = sp.misc.imresize(tmp, [self.ox, self.oy])
                scaled_tmp = (scaled_tmp - min(scaled_tmp.flatten()) ) / ( max(scaled_tmp.flatten()) - min(scaled_tmp.flatten()))
            else:
                scaled_tmp = np.zeros([self.ox,self.oy])
            input['tgt'][i] = scaled_tmp

        tmp = self.utils.gaussian(np.array([self.iy, self.ix]), input['loc'][:, -1], 4 * self.gauss)
        scaled_tmp = sp.misc.imresize(tmp, [self.ox, self.oy])
        scaled_tmp = (scaled_tmp - min(scaled_tmp.flatten())) / (max(scaled_tmp.flatten()) - min(scaled_tmp.flatten()))
        input['tgt'][self.nlandmark] = scaled_tmp

        return input
