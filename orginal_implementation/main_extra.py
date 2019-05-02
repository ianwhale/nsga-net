# main.py

import cv2
import torch
import random
import numpy as np
import os
import utils
from model import Model
from test import Tester
from train import Trainer
from config_extra import parser
from dataloader import Dataloader
import time
import pickle
from checkpoints import Checkpoints
from evolution import calculate_flops

def main():
    # load pop for extrapolation
    pop = pickle.load(open("Results/CIFAR10_baseline/Run5/pop_extra.pkl", "rb"))

    for i in range(len(pop)):
        genome = pop[i].genome
        # parse the arguments
        args = parser.parse_args()
        args.save = os.path.join("Extrapolation_results", "Model_ID_{}".format(pop[i].id))
        random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        utils.saveargs(args)

        # initialize the checkpoint class
        checkpoints = Checkpoints(args)

        # Create Model
        # genome = [[[0], [0, 0], [0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0, 0], [0]],
        #           [[0], [0, 0], [0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 1, 1], [0]],
        #           [[1], [0, 0], [0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0, 1], [1]]]
        # genome = [[[0], [0, 1], [1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1, 1], [0]],
        #           [[0], [0, 1], [0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1, 1], [0]],
        #           [[0], [0, 0], [0, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1, 1], [1]]]
        models = Model(args, genome)
        model, criterion, num_params = models.setup()
        model = calculate_flops.add_flops_counting_methods(model)
        # print(model)

        # Data Loading
        dataloader = Dataloader(args)
        loaders = dataloader.create()

        # The trainer handles the training loop
        trainer = Trainer(args, model, criterion)
        # The trainer handles the evaluation on validation set
        tester = Tester(args, model, criterion)

        # start training !!!
        loss_best = 1e10
        acc_test_list = []
        acc_best = 0
        for epoch in range(args.nepochs):

            # train for a single epoch
            start_time_epoch = time.time()
            if epoch == 0:
                model.start_flops_count()
            loss_train, acc_train = trainer.train(epoch, loaders)
            loss_test, acc_test = tester.test(epoch, loaders)
            acc_test_list.append(acc_test)

            if acc_test > acc_best:
                model_best = True
                # update the best test accu found so found
                acc_best = acc_test
                loss_best = loss_test
                checkpoints.save(epoch, model, model_best)

            time_elapsed = np.round((time.time() - start_time_epoch), 2)
            if epoch == 0:
                n_flops = (model.compute_average_flops_cost() / 1e6 / 2)

            if np.isnan(np.average(loss_train)):
                break

            print("Epoch {:d}:, test error={:0.2f}, FLOPs={:0.2f}M, n_params={:0.2f}M, {:0.2f} sec"
                  .format(epoch, 100.0-acc_test, n_flops, num_params/1e6, time_elapsed))

        # save the final model parameter
        # torch.save(model.state_dict(),
        #            "model_file/model%d.pth" % int(args.genome_id - 1))
        pop[i].fitness[0] = acc_best
        pop[i].fitness[1] = n_flops
        pop[i].n_params = num_params
        # error = 100 - np.mean(acc_test_list[-3:])

        # accuracy = acc_best
        # fitness = [acc_best, n_flops, num_params]
        # with open("output_file/output%d.pkl" % int(args.genome_id - 1), "wb") as f:
        #     pickle.dump(fitness, f)

    with open("Results/CIFAR10_baseline/Run5/pop_extra_evaluated.pkl", "wb") as f:
        pickle.dump(pop, f)

if __name__ == "__main__":
    main()
