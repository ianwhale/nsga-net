# monitor.py

from collections import OrderedDict


class Monitor:
    def __init__(self, smoothing=False, smoothness=0.7):
        self.keys = []
        self.losses = {}
        self.smoothing = smoothing
        self.smoothness = smoothness
        self.num = 0

    def register(self, modules):
        for m in modules:
            self.keys.append(m)
            self.losses[m] = 0

    def reset(self):
        self.num = 0
        for key, _ in self.losses.items():
            self.losses[key] = 0

    def update(self, modules, batch_size):
        if self.smoothing is False:
            for key, value in modules.items():
                self.losses[key] = (self.losses[key] * self.num + value * batch_size) / (self.num + batch_size)
        if self.smoothing is True:
            for key, value in modules.items():
                self.losses[key] = self.losses[key] * self.smoothness + value * (1 - self.smoothness)
        self.num += batch_size

    def getvalues(self, key=None):
        if key is not None:
            return self.losses[key]
        if key is None:
            return OrderedDict([(key,self.losses[key]) for key in self.keys])
