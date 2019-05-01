# logger.py

import os


class Logger:
    def __init__(self, path, filename):
        self.num = 0
        if os.path.isdir(path) is False:
            os.makedirs(path)
        self.filename = os.path.join(path, filename)
        self.fid = open(self.filename, 'w')
        self.fid.close()

    def register(self, modules):
        self.num = self.num + len(modules)
        tmpstr = ''
        for tmp in modules:
            tmpstr = tmpstr + tmp + '\t'
        tmpstr = tmpstr + '\n'
        self.fid = open(self.filename, 'a')
        self.fid.write(tmpstr)
        self.fid.close()

    def update(self, modules):
        tmpstr = ''
        for tmp in modules:
            tmpstr = tmpstr + '%.4f' % (modules[tmp]) + '\t'
        tmpstr = tmpstr + '\n'
        self.fid = open(self.filename, 'a')
        self.fid.write(tmpstr)
        self.fid.close()
