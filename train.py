# train.py

import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
# import lr_scheduler
import plugins


class Trainer():
    def __init__(self, args, model, criterion):
        self.args = args
        self.model = model
        self.criterion = criterion

        self.port = args.port
        self.dir_save = args.save

        self.cuda = args.cuda
        self.nepochs = args.nepochs
        self.nchannels = args.nchannels
        self.batch_size = args.batch_size
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide

        self.lr = args.learning_rate
        self.lr_min = args.min_learning_rate
        self.momentum = args.momentum
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.weight_decay = args.weight_decay
        self.optim_method = args.optim_method

        if self.optim_method == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        elif self.optim_method == 'RMSprop':
            self.optimizer = optim.RMSprop(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
        elif self.optim_method == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                momentum=self.momentum, nesterov=False
            )
        else:
            raise(Exception("Unknown Optimization Method"))

        # for classification
        self.label = torch.zeros(self.batch_size).long()
        self.input = torch.zeros(self.batch_size, self.nchannels,
                                 self.resolution_high, self.resolution_wide)

        if args.cuda:
            self.label = self.label.cuda()
            self.input = self.input.cuda()

        self.input = Variable(self.input)
        self.label = Variable(self.label)

        # logging training
        #self.log_loss_train = plugins.Logger(args.logs, 'TrainLogger.txt')
        self.params_loss_train = ['Loss', 'Accuracy']
        #self.log_loss_train.register(self.params_loss_train)

        # monitor training
        self.monitor_train = plugins.Monitor()
        self.params_monitor_train = ['Loss', 'Accuracy']
        self.monitor_train.register(self.params_monitor_train)

        # visualize training
        # self.visualizer_train = plugins.Visualizer(self.port, 'Train')
        # self.params_visualizer_train = {
        #     'Loss': {'dtype': 'scalar', 'vtype': 'plot'},
        #     'Accuracy': {'dtype': 'scalar', 'vtype': 'plot'},
        #     'Image': {'dtype': 'image', 'vtype': 'image'},
        #     'Images': {'dtype': 'images', 'vtype': 'images'},
        # }
        # self.visualizer_train.register(self.params_visualizer_train)

        # display training progress
        self.print_train = 'Train [%d/%d][%d/%d] '
        for item in self.params_loss_train:
            self.print_train = self.print_train + item + " %.4f "

        self.evalmodules = []
        self.losses_train = {}
        # print(self.model)

    def learning_rate(self, epoch):
        # training schedule
        return self.lr * (
            (0.1 ** int(epoch >= 60)) *
            (0.1 ** int(epoch >= 120)) *
            (0.1 ** int(epoch >= 160))
        )
    def cosine_annealing_lr(self, epoch):
        lr = self.lr_min + 0.5*(self.lr - self.lr_min)*(1 + np.cos(epoch/self.nepochs*np.pi))
        return lr

    # def auto_learning_rate(self, loss_history):
    #     # automatic training learning rate reduction
    #     threshold = 1000  # number of mini batches
    #     lr = self.lr
    #     if len(loss_history) > threshold:
    #         batches_without_decrease = lr_scheduler.count_steps_without_decrease(loss_history)
    #         print(batches_without_decrease)
    #         if batches_without_decrease > threshold:
    #             batches_without_decrease_robust = lr_scheduler.count_steps_without_decrease_robust(loss_history)
    #             print(batches_without_decrease_robust)
    #             if batches_without_decrease_robust > threshold:
    #                 lr = self.lr * 0.1
    #     return lr

    def get_optimizer(self, epoch, optimizer):
        # lr = self.auto_learning_rate(loss_history)
        lr = self.cosine_annealing_lr(epoch)
        # lr = self.learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def model_train(self):
        self.model.train()

    def train(self, epoch, dataloader):
        dataloader = dataloader['train']
        self.monitor_train.reset()
        data_iter = iter(dataloader)
        self.optimizer = self.get_optimizer(epoch + 1, self.optimizer)

        # switch to train mode
        self.model_train()

        i = 0
        acc_sum = 0
        loss_epoch = []
        while i < len(dataloader):

            ############################
            # Update network
            ############################

            input, label = data_iter.next()
            i += 1

            batch_size = input.size(0)
            self.input.data.resize_(input.size()).copy_(input)
            self.label.data.resize_(label.size()).copy_(label)

            output = self.model(self.input)
            loss = self.criterion(output, self.label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # this is for classfication
            pred = output.data.max(1)[1]
            acc = pred.eq(self.label.data).cpu().sum() * 100 / batch_size
            acc_sum = acc_sum + acc
            self.losses_train['Accuracy'] = acc
            self.losses_train['Loss'] = loss.data[0]
            self.monitor_train.update(self.losses_train, batch_size)

            # # print batch progress
            # print(self.print_train % tuple(
            #     [epoch, self.nepochs, i, len(dataloader)] +
            #     [self.losses_train[key] for key in self.params_monitor_train]))

            # append all mini-batches losses
            loss_epoch.append(loss.data[0])

        loss = self.monitor_train.getvalues()
        # self.log_loss_train.update(loss)
        loss['Image'] = input[0]
        loss['Images'] = input
        acc_avg = acc_sum / len(dataloader)
        # self.visualizer_train.update(loss)
        return self.monitor_train.getvalues('Loss'), acc_avg
