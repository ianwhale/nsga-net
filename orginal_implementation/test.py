# test.py

import torch
from torch.autograd import Variable

import plugins


class Tester():
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

        # for classification
        self.label = torch.zeros(self.batch_size).long()
        self.input = torch.zeros(self.batch_size, self.nchannels,
                                 self.resolution_high, self.resolution_wide)

        if args.cuda:
            self.label = self.label.cuda()
            self.input = self.input.cuda()

        self.input = Variable(self.input, volatile=True)
        self.label = Variable(self.label, volatile=True)

        # logging testing
        # self.log_loss_test = plugins.Logger(args.logs, 'TestLogger.txt')
        self.params_loss_test = ['Loss', 'Accuracy']
        # self.log_loss_test.register(self.params_loss_test)

        # monitor testing
        self.monitor_test = plugins.Monitor()
        self.params_monitor_test = ['Loss', 'Accuracy']
        self.monitor_test.register(self.params_monitor_test)

        # visualize testing
        # self.visualizer_test = plugins.Visualizer(self.port, 'Test')
        # self.params_visualizer_test = {
        #     'Loss': {'dtype': 'scalar', 'vtype': 'plot'},
        #     'Accuracy': {'dtype': 'scalar', 'vtype': 'plot'},
        #     'Image': {'dtype': 'image', 'vtype': 'image'},
        #     'Images': {'dtype': 'images', 'vtype': 'images'},
        # }
        # self.visualizer_test.register(self.params_visualizer_test)

        # display testing progress
        self.print_test = 'Test [%d/%d][%d/%d] '
        for item in self.params_loss_test:
            self.print_test = self.print_test + item + " %.4f "

        self.evalmodules = []
        self.losses_test = {}

    # not sure if this is working as it should
    # def model_eval(self):
    #     self.model.eval()
    #     for m in self.model.modules():
    #         for i in range(len(self.evalmodules)):
    #             if isinstance(m, self.evalmodules[i]):
    #                 m.train()

    def test(self, epoch, dataloader):
        dataloader = dataloader['test']
        self.monitor_test.reset()
        torch.cuda.empty_cache()
        data_iter = iter(dataloader)

        # switch to eval mode
        self.model.eval()

        i = 0
        acc_sum = 0
        while i < len(dataloader):

            ############################
            # Evaluate Network
            ############################

            input, label = data_iter.next()
            i += 1

            batch_size = input.size(0)
            self.input.data.resize_(input.size()).copy_(input)
            self.label.data.resize_(label.size()).copy_(label)

            self.model.zero_grad()
            output = self.model(self.input)
            loss = self.criterion(output, self.label)

            # this is for classification
            pred = output.data.max(1)[1]
            acc = pred.eq(self.label.data).cpu().sum() * 100 / batch_size
            acc_sum = acc_sum + acc
            self.losses_test['Accuracy'] = acc
            self.losses_test['Loss'] = loss.data[0]
            self.monitor_test.update(self.losses_test, batch_size)

            # print batch progress
            # print(self.print_test % tuple(
            #     [epoch, self.nepochs, i, len(dataloader)] +
            #     [self.losses_test[key] for key in self.params_monitor_test]))

        loss = self.monitor_test.getvalues()
        # self.log_loss_test.update(loss)
        loss['Image'] = input[0]
        loss['Images'] = input
        acc_avg = acc_sum/len(dataloader)
        # self.visualizer_test.update(loss)
        return self.monitor_test.getvalues('Loss'), acc_avg
