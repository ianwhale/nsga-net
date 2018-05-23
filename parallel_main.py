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
from config import parser
from dataloader import Dataloader
import time
import pickle
from scoop import futures
from checkpoints import Checkpoints
from evolution import calculate_flops

def main(pop, gpu_id):
    with torch.cuda.device(gpu_id):
        # parse the arguments
        args = parser.parse_args()
        random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        # utils.saveargs(args)

        # initialize the checkpoint class
        # checkpoints = Checkpoints(args)

        # Data Loading
        dataloader = Dataloader(args)
        loaders = dataloader.create()

        pop_fitness = []
        for i in range(len(pop)):
            print("Individual {}: on GPU {}".format(i, gpu_id))
            genome = pop[i].genome

            # Create Model
            models = Model(args, genome)
            model, criterion, num_params = models.setup(gpu_id)
            model = calculate_flops.add_flops_counting_methods(model)
            # print(model)

            # The trainer handles the training loop
            trainer = Trainer(args, model, criterion, gpu_id)
            # The trainer handles the evaluation on validation set
            tester = Tester(args, model, criterion, gpu_id)

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
                # if loss_best > loss_test:
                #     model_best = True
                #     loss_best = loss_test
                #     checkpoints.save(epoch, model, model_best)

                time_elapsed = np.round((time.time() - start_time_epoch), 2)
                if epoch == 0:
                    n_flops = (model.compute_average_flops_cost() / 1e6 / 2)
                # update the best test accu found so found
                if acc_test > acc_best:
                    acc_best = acc_test

                if np.isnan(np.average(loss_train)):
                    break

                print("Epoch {:d}:, test error={:0.2f}, FLOPs={:0.2f}M, n_params={:0.2f}M, {:0.2f} sec"
                      .format(epoch, 100.0-acc_test, n_flops, num_params/1e6, time_elapsed))

            # save the final model parameter
            # torch.save(model.state_dict(),
            #            "model_file/model%d.pth" % int(args.genome_id - 1))
            error = 100 - np.mean(acc_test_list[-3:])

            # accuracy = acc_best
            fitness = [error, n_flops, num_params, genome]

            pop_fitness.append(fitness)

    return pop_fitness

def split_pop(pop, ngpus):
    pop_eval = []
    pop_part_size = int(len(pop)/ngpus)
    complexity = []
    for p in pop:
        complexity.append(sum(p.phase_connection))
    idx_sort = np.argsort(complexity)
    pop_sorted = []
    for idx in idx_sort:
        pop_sorted.append(pop[idx])
    for i in range(ngpus):
        pop_part = []
        for j in range(i*pop_part_size, (i+1)*pop_part_size):
            pop_part.append(pop_sorted[j])
            # pop_part.append(pop[j])
        pop_eval.append(pop_part)
    return pop_eval

def parallel_evaulation(pop, ngpus):
    pop_fitness = list(futures.map(main, pop, range(ngpus)))
    pop_fitness_sorted = []
    for p in pop_fitness:
        pop_fitness_sorted.extend(p)
    return pop_fitness_sorted

if __name__ == "__main__":
    start = time.time()
    pop = pickle.load(open(os.path.join("input_file", "child.pkl"), "rb"))
    ngpus = 2
    pop_splited = split_pop(pop, ngpus)
    pop_fitness = parallel_evaulation(pop_splited, ngpus)
    for fitness in pop_fitness:
        for i in range(len(pop)):
            if fitness[-1] == pop[i].genome:
                pop[i].classification_error = fitness[0]
                pop[i].num_FLOPs = fitness[1]
                pop[i].num_params = fitness[2]
                pop[i].fitness[0] = pop[i].classification_error
                pop[i].fitness[1] = pop[i].num_FLOPs

    with open(os.path.join("output_file", "child.pkl"), "wb") as f:
        pickle.dump(pop, f)
    print(time.time() - start)
