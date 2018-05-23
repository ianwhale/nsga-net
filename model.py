# model.py

import math
import models
import losses
import pickle
import torch
from torch import nn
# from torch.nn import init
from models.model_utils import ParamCounter

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)  # this what to do
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()


class Model:

    def __init__(self, args, genome=[]):
        self.resume = args.resume
        self.ngpu = args.ngpu
        self.cuda = args.cuda
        self.nclasses = args.nclasses
        self.nfilters = args.nfilters
        self.nchannels = args.nchannels
        self.net_type = args.net_type
        self.genome_id = args.genome_id
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide
        self.resume = args.resume
        self.genome = genome
        if len(genome) < 1:
            with open("input_file/input%d.pkl" % int(self.genome_id - 1), "rb") as f:
                self.genome = pickle.load(f)

    def setup(self):
        # if self.net_type == 0:
        #     model = models.resnet18()
        #     print("Building ResNet18 model")
        # elif self.net_type == 1:
        #     model = models.PreActResNet18()
        #     print("Building PreActResNet18 model")
        # elif self.net_type == 2:
        #     model = models.VGG('VGG19')
        #     print("Building VGG19 model")
        # elif self.net_type == 3:
        #     model = models.GoogLeNet()
        #     print("Building GoogLeNet model")
        # elif self.net_type == 4:
        #     model = models.DenseNet121()
        #     print("Building DenseNet121 model")
        # elif self.net_type == 5:
        #     model = models.DPN92()
        #     print("Building CPN92 model")
        # elif self.net_type == 6:
        #     model = models.resnet20()
        #     print("Building ResNet 20 model")

        # model = getattr(models, self.net_type)(
        #     nchannels=self.nchannels,
        #     nfilters=self.nfilters,
        #     nclasses=self.nclasses,
        # )
        # model = models.net.Net()
        model = models.evonetwork.EvoNetwork(self.genome,
                                             [(self.nchannels, self.nfilters),
                                              (self.nfilters, self.nfilters),
                                              (self.nfilters, self.nfilters)],
                                             self.nclasses, (self.resolution_high,
                                                             self.resolution_wide),
                                             decoder="residual")
        # print(model)
        data = torch.randn(16, self.nchannels, self.resolution_high, self.resolution_wide)

        output = model(torch.autograd.Variable(data))
        num_params = ParamCounter(output).get_count()
        # print("Trainable parameters: {}".format(num_params))


        criterion = losses.Classification()

        if self.cuda:
            model = nn.DataParallel(model, device_ids=list(range(self.ngpu)))
            model = model.cuda()
            criterion = criterion.cuda()

        model_path = "model_file/model0.pth"
        if self.resume:
            model.load_state_dict(torch.load(model_path))
        else:
            model.apply(weights_init)

        # if self.resume is None:
        #     model.apply(weights_init)
        # else:
        #     # load the model parameter
        #     model.load_state_dict(torch.load(self.resume))

        return model, criterion, num_params
