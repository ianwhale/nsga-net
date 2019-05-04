'''Inference CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
# update your projecty root path before running
sys.path.insert(0, 'path/to/nsga-net')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision

import os
import sys
import time
import logging
import argparse
import numpy as np

from misc import utils

# model imports
from models import macro_genotypes
from models.macro_models import EvoNetwork
import models.micro_genotypes as genotypes
from models.micro_models import PyramidNetworkCIFAR as PyrmNASNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--layers', default=20, type=int, help='total number of layers (default: 20)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--arch', type=str, default='NSGANet', help='which architecture to use')
parser.add_argument('--filter_increment', default=4, type=int, help='# of filter increment')
parser.add_argument('--SE', action='store_true', default=False, help='use Squeeze-and-Excitation')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--net_type', type=str, default='micro', help='(options)micro, macro')

args = parser.parse_args()

args.save = 'infer-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

device = 'cuda'

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.auxiliary and args.net_type == 'macro':
        logging.info('auxiliary head classifier not supported for macro search space models')
        sys.exit(1)

    logging.info("args = %s", args)

    cudnn.enabled = True
    cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Data
    _, valid_transform = utils._data_transforms_cifar10(args)

    valid_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    if args.net_type == 'micro':
        logging.info("==> Building micro search space encoded architectures")
        genotype = eval("genotypes.%s" % args.arch)
        net = PyrmNASNet(args.init_channels, num_classes=10, layers=args.layers,
                         auxiliary=args.auxiliary, genotype=genotype,
                         increment=args.filter_increment, SE=args.SE)
    elif args.net_type == 'macro':
        genome = eval("macro_genotypes.%s" % args.arch)
        channels = [(3, 128), (128, 128), (128, 128)]
        net = EvoNetwork(genome, channels, 10, (32, 32), decoder='dense')
    else:
        raise NameError('Unknown network type, please only use supported network type')

    # logging.info("{}".format(net))
    logging.info("param size = %fMB", utils.count_parameters_in_MB(net))

    net = net.to(device)
    # no drop path during inference
    net.droprate = 0.0
    utils.load(net, args.model_path)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # inference on original CIFAR-10 test images
    infer(valid_queue, net, criterion)


def infer(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    logging.info('valid acc %f', acc)

    return test_loss/total, acc


if __name__ == '__main__':
    main()

