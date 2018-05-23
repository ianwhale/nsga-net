# loaders.py

import torch
import numpy as np
from PIL import Image

def loader_image(path):
    return Image.open(path).convert('RGB')

def loader_torch(path):
    return torch.load(path)

def loader_numpy(path):
    return np.load(path)