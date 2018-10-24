# config file for running nsga2
import os
import datetime
import argparse


def parse_args():
    result_path = "results/"
    result_path = os.path.join(result_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))

    parser = argparse.ArgumentParser(description='Deep Neural Network Architecture Search using EC')

    # ======================= EC related Parameters =====================================
    parser.add_argument('--pop-size', type=int, default=10, help='population size')
    parser.add_argument('--max-gen-exploration', type=int, default=20,
                        help='maximum number of generations')
    parser.add_argument('--max-gen-exploitation', type=int, default=10,
                        help='maximum number of generations')
    parser.add_argument('--n-obj', type=int, default=2, help='number of objectives (1-3)')
    parser.add_argument('--n-constr', type=int, default=0, help='number of constraints')
    parser.add_argument('--lb', nargs='+', type=float, default=None, help='lower variable bound')
    parser.add_argument('--ub', nargs='+', type=float, default=None, help='lower variable bound')
    parser.add_argument('--p-xover', type=float, default=1.0, help='crossover probability')
    parser.add_argument('--p-mut', type=float, default=0.00, help='mutation probability')
    parser.add_argument('--eta-xover', type=int, default=20, help='crossover distribution index')
    parser.add_argument('--eta-mut', type=int, default=15, help='mutation distribution index')
    parser.add_argument('--run', type=int, default=0, help='the run id')
    parser.add_argument('--EPS', type=float, default=1.0e-14, help='constant value')
    parser.add_argument('--seed', type=int, default=10, help='random number seed')
    parser.add_argument('--prior-knowledge', type=bool, default=False,
                        help='seed initial population with prior knowledge')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume from previous generation')
    parser.add_argument('--resume-gen', type=int, default=23,
                        help='Resume NSGA2 from this specific generation')
    parser.add_argument('--results_dir', type=str, default="CIFAR10_offspring",
                        help='Resume NSGA2 from this specific generation')
    parser.add_argument('--term', type=bool, default=False,
                        help='Whether or not to check for early termination')
    parser.add_argument('--term-threshold', type=float, default=0.05,
                        help='Whether or not to check for early termination')
    parser.add_argument('--eval-method', type=int, default=2,
                        help='0: non-cluster, 1: cluster, 2: dummy')

    # ======================= EDNN related Parameters =====================================
    parser.add_argument('--n-phases', type=int, default=3, help='number of phases')
    parser.add_argument('--n-nodes', type=int, default=6, help='number of nodes per phases')
    parser.add_argument('--bprob-time', type=int, default=1800,
                        help='estimated time for one back propagation training in seconds')

    # ======================== Data Setings ============================================
    parser.add_argument('--dataset-test', type=str, default='MYCIFAR10', help='name of training dataset')
    parser.add_argument('--dataset-train', type=str, default='MYCIFAR10', help='name of training dataset')
    parser.add_argument('--split_test', type=float, default=None, help='test split')
    parser.add_argument('--split_train', type=float, default=None, help='train split')
    parser.add_argument('--dataroot', type=str, default='../', help='path to the data')
    parser.add_argument('--save', type=str, default=result_path +'Save', help='save the trained models here')
    parser.add_argument('--logs', type=str, default=result_path +'Logs', help='save the training log files here')
    parser.add_argument('--bp_resume', type=bool, default=False, help='full path of models to resume training')
    parser.add_argument('--nclasses', type=int, default=10, help='number of classes for classification')
    parser.add_argument('--input-filename-test', type=str, default=None, help='input test filename for filelist and folderlist')
    parser.add_argument('--label-filename-test', type=str, default=None, help='label test filename for filelist and folderlist')
    parser.add_argument('--input-filename-train', type=str, default=None, help='input train filename for filelist and folderlist')
    parser.add_argument('--label-filename-train', type=str, default=None, help='label train filename for filelist and folderlist')
    parser.add_argument('--loader-input', type=str, default=None, help='input loader')
    parser.add_argument('--loader-label', type=str, default=None, help='label loader')

    # ======================== Network Model Setings ===================================
    parser.add_argument('--nchannels', type=int, default=3, help='number of input channels')
    parser.add_argument('--nfilters', type=int, default=64, help='number of filters in conv layer')
    parser.add_argument('--resolution-high', type=int, default=32, help='image resolution height')
    parser.add_argument('--resolution-wide', type=int, default=32, help='image resolution width')
    parser.add_argument('--ndim', type=int, default=None, help='number of feature dimensions')
    parser.add_argument('--nunits', type=int, default=None, help='number of units in hidden layers')
    parser.add_argument('--dropout', type=float, default=None, help='dropout parameter')
    # parser.add_argument('--net-type', type=str, default='resnet18', help='type of network')
    parser.add_argument('--net-type', type=int, default=0, help='type of network')
    parser.add_argument('--length-scale', type=float, default=None, help='length scale')
    parser.add_argument('--tau', type=float, default=None, help='Tau')
    parser.add_argument('--genome_id', type=int, default=None, metavar='', help='none')

    # ======================== Training Settings =======================================
    parser.add_argument('--cuda', type=bool, default=True, help='run on gpu')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training')
    parser.add_argument('--nepochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--niters', type=int, default=None, help='number of iterations at test time')
    parser.add_argument('--epoch-number', type=int, default=None, help='epoch number')
    parser.add_argument('--nthreads', type=int, default=12, help='number of threads for data loading')
    parser.add_argument('--manual-seed', type=int, default=0, help='manual seed for randomness')
    parser.add_argument('--port', type=int, default=None, help='port for visualizing training at http://localhost:port')

    # ======================== Hyperparameter Setings ==================================
    parser.add_argument('--optim-method', type=str, default='SGD', help='the optimization routine ')
    parser.add_argument('--learning-rate', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--min-learning-rate', type=float, default=1e-8, help='learning rate')
    parser.add_argument('--learning-rate-decay', type=float, default=None, help='learning rate decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--adam-beta1', type=float, default=0.9, help='Beta 1 parameter for Adam')
    parser.add_argument('--adam-beta2', type=float, default=0.999, help='Beta 2 parameter for Adam')

    args = parser.parse_args()
    return args
