# -*- coding: utf-8 -*-
"""
Created on Sat March 03

@author: Zhichao Lu
"""
import os
import time
import shutil
import pickle
import itertools
import nsga2_config
import nsga2_individual
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plugins.genome_visualizer import make_dot_phase
from plugins.genome_visualizer import make_dot_genome
#pytorchnet related
import torch
import subprocess
from model import Model
from test import Tester
from train import Trainer
from dataloader import Dataloader
from evolution import calculate_flops
# from evolution.check_duplicates import overall_check
import evolution.connectivity_matrix as conn_matrix

class Engine:
    def __init__(self, args, run=0):
        self.run = run
        self.seed = args.seed
        self.resume = args.resume
        self.resume_gen = args.resume_gen
        self.pop_size = args.pop_size
        self.n_obj = args.n_obj
        self.n_constr = args.n_constr
        self.lb = args.lb
        self.ub = args.ub
        self.p_xover = args.p_xover
        self.p_mut = args.p_mut
        self.EPS = args.EPS
        self.n_phases = args.n_phases
        self.n_nodes = args.n_nodes
        self.phase_length = (args.n_nodes - 1)*args.n_nodes/2 + 1
        self.bprob_time = args.bprob_time
        self.prior_knowledge = args.prior_knowledge
        self.results_dir = args.results_dir
        self.eval_method = args.eval_method

    def initialization(self):
        # two stages initialization
        config_archive = []  # initialize a empty unique config list
        # 1st stage: extract different unique phase configurations
        phase_length = int((self.n_nodes-1)*self.n_nodes/2 + 1)
        possible_phase_string = list(map(list, itertools.product([0, 1], repeat=phase_length)))
        # randomly shuffle the list
        np.random.shuffle(possible_phase_string)

        for phase_string in possible_phase_string:
            config_archive = self.update_config_archive(phase_string, config_archive, self.n_nodes)
            # terminate the loop if the desired number of unique phase configuration is reached
            if len(config_archive) >= self.pop_size*self.n_phases:
                break

        # 2nd stage: assemble these phase config together
        pop = []
        idx = np.random.permutation(len(config_archive))
        # check if found phase config is enough for creating initial population
        while len(idx) < self.pop_size*self.n_phases:
            idx = np.append(idx, np.random.permutation(len(config_archive)))

        idx = idx[:self.pop_size*self.n_phases]
        idx = idx.tolist()

        for i in range(0, len(idx), self.n_phases):
            bit_string, conn = [], []
            for j in range(self.n_phases):
                conn.append(idx[i+j])
                bit_string.append(config_archive[idx[i+j]].bit_string)
            pop.append(nsga2_individual.IndividualEDNN(bit_string, self.n_phases, self.n_nodes))
            pop[-1].phase_connection = conn
            pop[-1].pt = 1
            pop[-1].id = "{}_{}".format(0, len(pop)-1)

        return pop, config_archive

    def resume_from_previous(self):
        file_dir = os.path.join("Results", self.results_dir,
                                "Run{}/Gen{}".format(self.run, self.resume_gen))
        with open(os.path.join(file_dir, "pop.pkl"), "rb") as f:
            pop = pickle.load(f)
        with open(os.path.join(file_dir, "config_archive.pkl"), "rb") as f:
            config_archive = pickle.load(f)
        with open(os.path.join(file_dir, "pop_archive.pkl"), "rb") as f:
            pop_archive = pickle.load(f)

        return pop, config_archive, pop_archive

    def seed_prior_knowledge(self):
        # seed the initial population with these good solution candidates
        if self.n_nodes == 4:
            vgg_string = [1, 0, 1, 0, 0, 1, 0]
            resnet_string = [1, 0, 1, 0, 0, 1, 1]
            densenet_string = [1, 1, 1, 1, 1, 1, 0]
        elif self.n_nodes == 6:
            vgg_string = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
            resnet_string = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
            densenet_string = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        else:
            vgg_string, resnet_string, densenet_string = [], [], []

        return [resnet_string, densenet_string, vgg_string]

    def report_config_to_file(self, nsga2_config):
        dir_name = "Results"
        if os.path.exists(dir_name):
            pass
        else:
            os.makedirs(dir_name)

        dir_name = os.path.join(dir_name, self.results_dir)
        if os.path.exists(dir_name):
            pass
        else:
            os.makedirs(dir_name)

        file_name = os.path.join(dir_name, "configurations.txt")
        with open(file_name, "w") as text_file:
            print("NSGA2 related:", file=text_file)
            print("Seed = {}, pop size = {}, max gens = {}, resume = {}, prior = {}"
                  .format(nsga2_config.seed, nsga2_config.pop_size,
                          nsga2_config.max_gen_exploration, nsga2_config.resume,
                          nsga2_config.prior_knowledge), file=text_file)
            print("Crx prob = {}, mut prob = {}, termination = {}"
                  .format(nsga2_config.p_xover, nsga2_config.p_mut,
                          nsga2_config.term),
                  file=text_file)
            print("Pytorch related:", file=text_file)
            print("Num phases = {}, num nodes = {}".format(self.n_phases, self.n_nodes),
                  file=text_file)
            print("Dataset = {}, seed = {}, num filters = {}, batch size = {}, num epochs = {}"
                  .format(nsga2_config.dataset_train, nsga2_config.manual_seed,
                          nsga2_config.nfilters, nsga2_config.batch_size,
                          nsga2_config.nepochs), file=text_file)
            print("Train = {}, lr = {}".format(nsga2_config.optim_method,
                                               nsga2_config.learning_rate),
                  file=text_file)

        return

    @staticmethod
    def update_config_archive(phase_string, config_archive, n_nodes):
        if all(bit == 0 for bit in phase_string[:-1]):
            pass
        else:
            # initiate a phase object
            config = nsga2_individual.Phase(phase_string)
            # assume we have not seen this config before
            not_seen = True
            for i in range(len(config_archive)):
                if conn_matrix.duplicate_check(config.bit_string[:-1],
                                               config_archive[i].bit_string[:-1], n_nodes):
                    # connection matrix check does not differentiate the last bit
                    if config.bit_string[:-1] != config_archive[i].bit_string[:-1]:
                        break
                    # check to verify if already included in equivalent phases
                    already_included = False
                    for member in config_archive[i].equivalent_phases:
                        if config.key == member.key:
                            already_included = True
                            break
                    if not already_included:
                        config_archive[i].append_equivalent_phase(config)
                    not_seen = False
                    break

            if not_seen:
                config.id = len(config_archive) + 1
                config_archive.append(config)

        return config_archive

    @staticmethod
    def update_pop_archive(pop, pop_archive, recursive=False):
        if recursive:
            for p in pop:
                not_seen = True
                for i in range(len(pop_archive)):
                    if p.phase_connection == pop_archive[i].phase_connection:
                        pop_archive[i].fitness = p.fitness
                        not_seen = False
                        break
                if not_seen:
                    pop_archive.append(p)
        else:
            pop_archive.extend(pop)

        return pop_archive

    @staticmethod
    def assign_conn(indv, config_archive):
        conn = []
        for phase in indv.phases:
            phase_found = False
            for i in range(len(config_archive)):
                if phase.key == config_archive[i].key:
                    phase_found = True
                else:
                    for j in range(len(config_archive[i].equivalent_phases)):
                        if phase.key == config_archive[i].equivalent_phases[j].key:
                            phase_found = True
                            break
                if phase_found:
                    conn.append(i)
                    break
        return conn

    def compute_fitness_pop(self, pop, pop_archive, args_pytorchnet):
        # compute fitness
        self.assign_fitness(pop, args_pytorchnet)
        # update population archive
        pop_archive = self.update_pop_archive(pop, pop_archive)

        return pop, pop_archive

    def assign_fitness(self, pop, args_pytorchnet):
        #  if the population to be evaluated are empty
        if len(pop) < 1:
            return

        if self.eval_method == 2:
            # dummy fitness assignment for debug purpose
            for i in range(len(pop)):
                pop[i].fitness[0] = 0
                pop[i].fitness[1] = sum(pop[i].active_nodes)
        elif self.eval_method == 1:
            # PBS cluster based parallel Pytorchnet
            self.cluster_pytorchnet(pop)
        else:
            # non-cluster version of using Pytorchnet
            self.non_cluster_pytorchnet(pop, args_pytorchnet)

        return

    def cluster_pytorchnet(self, pop):
        # 1st prepare the input, output and model file directory
        self.prepare_IO_folders()

        # prepare the input file for parallel evaluation
        for i in range(len(pop)):
            with open("input_file/input%d.pkl" % i, "wb") as f:
                pickle.dump(pop[i].genome, f)

        # run parallel evaluation in HPCC - qsub bash commend
        subprocess.call("qsub evaluate_pytorchnet.qsub", shell=True)

        # wait for the parallel execution to finish
        time.sleep(self.bprob_time)  # pause for estimated back-prob time

        while True:
            time.sleep(300)  # check every 5 minutes
            n_completed = 0
            for k in range(len(pop)):
                if os.path.isfile("output_file/output%d.pkl" % k):
                    n_completed += 1

            if n_completed >= len(pop):
                break

        # hpcc parallel evaluation finished
        for i in range(len(pop)):
            file_name = "output_file/output{}.pkl".format(i)
            if os.path.isfile(file_name):
                with open(file_name, "rb") as f:
                    temp = pickle.load(f)
                pop[i].fitness[0] = temp[0]
                pop[i].fitness[1] = temp[1]
                pop[i].n_params = temp[2]
            else:
                pop[i].fitness = [100, 1e5]

        return

    @staticmethod
    def non_cluster_pytorchnet(pop, args):
        for i in range(len(pop)):
            torch.manual_seed(args.manual_seed)

            # Create Model
            models = Model(args, pop[i].genome)
            model, criterion, num_params = models.setup()
            model = calculate_flops.add_flops_counting_methods(model)

            # Data Loading
            dataloader = Dataloader(args)
            loaders = dataloader.create()

            # The trainer handles the training loop
            trainer = Trainer(args, model, criterion)
            # The trainer handles the evaluation on validation set
            tester = Tester(args, model, criterion)

            # start training !!!
            acc_test_list = []
            acc_best = 0
            train_time_start = time.time()
            for epoch in range(args.nepochs):
                # train for a single epoch

                if epoch == 0:
                    model.start_flops_count()
                loss_train, acc_train = trainer.train(epoch, loaders)
                loss_test, acc_test = tester.test(epoch, loaders)
                acc_test_list.append(acc_test)

                if epoch == 0:
                    n_flops = (model.compute_average_flops_cost() / 1e6 / 2)
                # update the best test accu found so found
                if acc_test > acc_best:
                    acc_best = acc_test

                # print("Epoch {}, train loss = {}, test accu = {}, best accu = {}, {} sec"
                #       .format(epoch, np.average(loss_train), acc_test, acc_best, time_elapsed))
                
                if np.isnan(np.average(loss_train)):
                    break

            # end of training
            time_elapsed = np.round((time.time() - train_time_start), 2)
            pop[i].fitness[0] = 100.0-np.mean(acc_test_list[-3:])
            pop[i].fitness[1] = n_flops
            pop[i].n_params = num_params
            pop[i].n_FLOPs = n_flops
            print("Indv {:d}:, test error={:0.2f}, FLOPs={:0.2f}M, n_params={:0.2f}M, {:0.2f} sec"
                  .format(i, pop[i].fitness[0], n_flops, num_params/1e6, time_elapsed))

        return

    def assign_config_fitness(self, config_archive, pop_archive):
        for i in range(len(config_archive)):
            error, occur, freq = [0, 0, 0], [0, 0, 0], [0, 0, 0]
            for p in pop_archive:
                for k in range(self.n_phases):
                    if i == p.phase_connection[k]:
                        error[k] += p.fitness[0]
                        occur[k] += 1
            for ph in range(self.n_phases):
                if occur[ph] > 0:
                    error[ph] = error[ph]/occur[ph]
                    freq[ph] = occur[ph]/len(pop_archive)
                else:
                    error[ph] = np.nan
            config_archive[i].fitness = error
            config_archive[i].frequency = freq

        return

    def process_config(self, pop, config_archive):
        # 1st check if any new phase config is found in pop
        for indv in pop:
            for phase in indv.phases:
                config_archive = self.update_config_archive(phase.bit_string,
                                                            config_archive, self.n_nodes)

        # 2nd assign the conn for each individual
        for i in range(len(pop)):
            conn = self.assign_conn(pop[i], config_archive)
            pop[i].phase_connection = conn

        return pop, config_archive

    def prepare_pop_for_evaluation(self, gen, pop, config_archive, pop_archive, exploitation=False):
        # update phase config archive in case new config is found from pop
        pop, config_archive = self.process_config(pop, config_archive)

        pop_eval = []  # unique individuals

        for i in range(len(pop)):
            duplicates = False  # assume this individual is not duplicate to begin with
            # check against current pop for duplicates
            for j in range(i):
                if pop[i].phase_connection == pop[j].phase_connection:
                    duplicates = True
                    break
            # check against pop archive for duplicates
            if not duplicates:
                for member in pop_archive:
                    if pop[i].phase_connection == member.phase_connection:
                        duplicates = True
                        break
            if not duplicates:
                pop_eval.append(pop[i])

        # in case there is some duplicates in pop after genetic operation
        if exploitation:
            pop_eval = self.apply_heuristics(pop_eval, config_archive, pop_archive)

        for i in range(len(pop_eval)):
            pop_eval[i].id = "{}_{}".format(gen, i)

        return pop_eval, config_archive

    def apply_heuristics(self, pop, config_archive, pop_archive):
        # pop = self.heuristic_elitism_recombination(pop, config_archive, pop_archive)
        pop = self.heuristic_recombination(pop, config_archive, pop_archive)

        return pop

    def sample_conn_from_bayesian(self, pop_archive):
        conn = []
        for ph in range(self.n_phases):
            if ph == 0:
                idx = np.random.randint(len(pop_archive))
            else:
                dependencies = []
                for i in range(len(pop_archive)):
                    if pop_archive[i].phase_connection[ph - 1] == conn[ph - 1]:
                        dependencies.append(i)
                idx = np.random.choice(dependencies)

            conn.append(pop_archive[idx].phase_connection[ph])

        return conn

    def heuristic_recombination(self, pop, config_archive, pop_archive):
        if len(pop) < self.pop_size:
            max_trail = 20  # set the maximum trails to avoid infinite loop
            pop_bayesian = self.fill_nondominated_sort(pop_archive, int(len(pop_archive) / 2))

            for i in range(len(pop), self.pop_size):
                trial = 1
                while True:
                    conn = self.sample_conn_from_bayesian(pop_bayesian)
                    duplicate = False  # assumes this conn is not duplicate
                    # 1st check if conn exists in current population
                    for p in pop:
                        if p.phase_connection == conn:
                            duplicate = True
                            break
                    # 2nd check if conn exists in population archive
                    for member in pop_archive:
                        if member.phase_connection == conn:
                            duplicate = True
                            break
                    if (not duplicate) or (trial > max_trail):
                        break
                    # print(trial)
                    trial += 1

                if not duplicate:
                    bit_string = []
                    for phase in range(len(conn)):
                        # for c in conn:
                        bit_string.append(config_archive[conn[phase]].bit_string)
                    pop.append(nsga2_individual.IndividualEDNN(bit_string, self.n_phases, self.n_nodes))
                    pop[-1].phase_connection = conn
                    trial = 0
                    pop[-1].cr = 0

        return pop

    @staticmethod
    def prepare_IO_folders():
        # 1st delete the input_file directory w/ all contents
        if os.path.exists("input_file"):
            shutil.rmtree("input_file")
            os.makedirs("input_file")
        else:
            os.makedirs("input_file")

        # 2nd delete the output_file directory w/ all contents
        if os.path.exists("output_file"):
            shutil.rmtree("output_file")
            os.makedirs("output_file")
        else:
            os.makedirs("output_file")

        # 3rd delete the model_file directory w/ all contents
        if os.path.exists("model_file"):
            shutil.rmtree("model_file")
            os.makedirs("model_file")
        else:
            os.makedirs("model_file")

    def report_gen_to_file(self, run, gen, pop, child, config_archive, pop_archive):
        dir_name = "Results"
        if os.path.exists(dir_name):
            pass
        else:
            os.makedirs(dir_name)

        dir_name = os.path.join(dir_name, self.results_dir)
        if os.path.exists(dir_name):
            pass
        else:
            os.makedirs(dir_name)

        dir_name = os.path.join(dir_name, "Run{}".format(run))
        if os.path.exists(dir_name):
            pass
        else:
            os.makedirs(dir_name)

        dir_name = os.path.join(dir_name, "Gen{}".format(gen))
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
        else:
            os.makedirs(dir_name)

        # assign a unique ID number to each parent individual
        for i in range(len(pop)):
            if len(pop[i].id) < 1:
                pop[i].id = "{}_{}_p".format(gen, i)

        file_name = os.path.join(dir_name, "pop.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(pop, f)

        if len(child) > 0:
            file_name = os.path.join(dir_name, "child.pkl")
            with open(file_name, "wb") as f:
                pickle.dump(child, f)

        file_name = os.path.join(dir_name, "config_archive.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(config_archive, f)

        file_name = os.path.join(dir_name, "pop_archive.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(pop_archive, f)

        # mark every one in the survived population as parent
        for i in range(len(pop)):
            pop[i].pt = 1

        return pop

    def dominates(self, p, q):
        # ...check if p dominates q
        # ..returns 1 of p dominates q, -1 if q dominates p, 0 if both are non-dominated
        flag1 = 0
        flag2 = 0
        for i in range(self.n_obj):
            if p.fitness[i] < q.fitness[i]:
                flag1 = 1
            else:
                if p.fitness[i] > q.fitness[i]:
                    flag2 = 1

        if (flag1 == 1) and (flag2 == 0):
            return 1
        else:
            if (flag1 == 0) and (flag2 == 1):
                return -1
            else:
                return 0

    def tour_select(self, ind1, ind2):
        # print 'doing binary tournament selection'
        dom_1_2 = self.dominates(ind1, ind2)
        if dom_1_2 == 1:
            return ind1
        elif dom_1_2 == -1:
            return ind2
        else:
            if ind1.crowding_dist > ind2.crowding_dist:
                return ind1
            elif ind2.crowding_dist > ind1.crowding_dist:
                return ind2
            elif np.random.rand() > 0.5:
                return ind1
            else:
                return ind2

    def crossover(self, parent1, parent2):
        p1_bit_string = np.concatenate([np.array(i) for i in parent1.bit_string])
        p2_bit_string = np.concatenate([np.array(i) for i in parent2.bit_string])
        common_bits = (p1_bit_string-p2_bit_string) == 0
        child_base = np.zeros((len(p1_bit_string),), dtype=np.int)
        child_base[common_bits] = p1_bit_string[common_bits]
        compl = np.array([np.sum(p1_bit_string), np.sum(p2_bit_string)]) - np.sum(child_base)
        compl_lb, compl_ub = np.amin(compl), np.amax(compl)
        # assuming create two offspring from two parents
        offspring = []
        for i in range(2):
            # 1 blend of two parents
            if np.random.rand() < 1.5:
                child_bit_string = np.copy(child_base)
                idx, = np.where(common_bits == False)
                idx = np.random.permutation(idx)
                if compl_ub > compl_lb:
                    idx = idx[:np.random.randint(compl_lb, compl_ub)]
                else:
                    idx = idx[:compl_lb]
                child_bit_string[idx] = 1
                child_bit_string = self.convert_to_bit_string(child_bit_string)
            # 2 swap phases
            else:
                child_bit_string = []
                for j in range(self.n_phases):
                    if np.random.rand() < 0.5:
                        child_bit_string.append(parent1.bit_string[j])
                    else:
                        child_bit_string.append(parent2.bit_string[j])

            offspring.append(nsga2_individual.IndividualEDNN(child_bit_string,
                                                             self.n_phases, self.n_nodes))
            # fill in details
            offspring[-1].cr = 1
            offspring[-1].parents.append(parent1)
            offspring[-1].parents.append(parent2)

        return offspring

    def convert_to_bit_string(self, bit_string):
        # put a list into list of phase list
        list_of_phase_list = []
        phase_list = []
        for bit in bit_string:
            phase_list.append(bit)
            if len(phase_list) >= self.phase_length:
                # in case an entire phase is empty
                if all(v == 0 for v in phase_list[:-1]):
                    phase_list[0] = 1
                list_of_phase_list.append(phase_list)
                phase_list = []

        return list_of_phase_list

    def selection(self, old_pop):
        if np.random.rand() < self.p_xover:
            a1 = np.random.permutation(range(self.pop_size))
            a2 = np.random.permutation(range(self.pop_size))
            a = np.append(a1, a2)
            new_pop = []
            for i in range(0, 2*self.pop_size, 4):
                parent1 = self.tour_select(old_pop[a[i]], old_pop[a[i + 1]])
                parent2 = self.tour_select(old_pop[a[i + 2]], old_pop[a[i + 3]])
                new_pop.extend(self.crossover(parent1, parent2))
        else:
            new_pop = old_pop
        # # for debug purpose:
        # print("Offspring:")
        # for indv in new_pop:
        #     indv.display_content()
        return new_pop

    def mutation(self, old_pop, gen):
        new_pop = []
        # three different mutation method
        for p in old_pop:
            mut_option = np.random.randint(1, 2)
            if mut_option == 1:
                # first mutation method: regular binary mutation
                bit_string, mutated = self.mutation_bin(p)
            elif mut_option == 2:
                # second mutation method: shuffle the indv phases order
                bit_string, mutated = self.mutation_shuffle(p)

            new_pop.append(nsga2_individual.IndividualEDNN(bit_string, self.n_phases, self.n_nodes))
            new_pop[-1].id = "{}_{}".format(gen, len(new_pop)-1)

            if mutated:
                new_pop[-1].mut_option = mut_option
                new_pop[-1].parents.append(p)
            else:
                for parent in p.parents:
                    new_pop[-1].parents.append(parent)

        return new_pop

    def mutation_bin(self, indv):
        bit_string = []
        mutated = False
        for bit in np.concatenate([np.array(i) for i in indv.bit_string]):
            if np.random.rand() < self.p_mut:
                mutated = True
                if bit > 0:
                    bit_string.append(0)
                else:
                    bit_string.append(1)
            else:
                bit_string.append(bit)

        return self.convert_to_bit_string(bit_string), mutated

    def mutation_shuffle(self, indv):
        mutated = False
        if np.random.rand() < self.p_mut:
            mutated = True
            a = np.random.permutation(self.n_phases)
            bit_string = []
            for i in range(self.n_phases):
                bit_string.append(indv.bit_string[a[i]])
        else:
            bit_string = indv.bit_string[:]

        return bit_string, mutated

    def sort_wrt_obj(self, f, j):
        obj_values = []
        for p in f:
            obj_values.append(p.fitness[j])

        r = range(len(f))
        g = [x for _, x in sorted(zip(obj_values, r))]

        return g, max(obj_values), min(obj_values)

    def crowding_distance(self, F):
        # ....
        for f in F:
            f_len = len(f)
            if (f_len <= 2):
                for p in f:
                    p.crowding_dist = float('inf')
            else:
                for p in f:
                    p.crowding_dist = 0.0

                for j in range(self.n_obj):
                    f_id, max_j, min_j = self.sort_wrt_obj(f, j)
                    f[f_id[0]].crowding_dist = float("inf")
                    # f[f_id[-1]].crowding_dist = float("inf")
                    if max_j != min_j:
                        for i in range(1, f_len - 1):
                            f[f_id[i]].crowding_dist = f[f_id[i]].crowding_dist + \
                                                       (f[f_id[i + 1]].fitness[j] - f[f_id[i - 1]].fitness[j]) / (
                                                                   max_j - min_j)

        return

    def assign_rank_crowding_distance(self, pop):
        F = []  # ...list of fronts...
        f = []

        for p in pop:
            p.S_dom = []
            p.n_dom = 0
            for q in pop:
                dom_p_q = self.dominates(p, q)
                if (dom_p_q == 1):
                    p.S_dom.append(q)
                elif (dom_p_q == -1):
                    p.n_dom = p.n_dom + 1

            if p.n_dom == 0:
                p.rank = 1
                f.append(p)

        F.append(f)
        i = 0
        while (len(F[i]) != 0):
            Q = []
            for p in F[i]:
                for q in p.S_dom:
                    q.n_dom = q.n_dom - 1
                    if q.n_dom == 0:
                        q.rank = i + 2
                        Q.append(q)
            i = i + 1
            F.append(Q)

        if len(F[-1]) == 0:
            del (F[-1])

        self.crowding_distance(F)

        return F

    def sort_wrt_crowding_dist(self, f):
        c_dist_vals = []
        for p in f:
            c_dist_vals.append(p.crowding_dist)

        r = range(len(f))
        f_new_id = [x for _, x in sorted(zip(c_dist_vals, r), reverse=True)]

        return f_new_id

    def fill_nondominated_sort(self, mixed_pop, pop_size=[]):
        filtered_pop = []
        if not pop_size:
            pop_size = self.pop_size

        selected_fronts = self.assign_rank_crowding_distance(mixed_pop)
        counter = 0
        candidate_fronts = []
        for f in selected_fronts:
            candidate_fronts.append(f)
            counter += len(f)
            if counter > pop_size:
                break

        n_fronts = len(candidate_fronts)

        if n_fronts == 1:
            filtered_pop = []
        else:
            for i in range(n_fronts - 1):
                filtered_pop.extend(candidate_fronts[i])

        n_pop_curr = len(filtered_pop)

        sorted_final_front_id = self.sort_wrt_crowding_dist(candidate_fronts[-1])

        for i in range(pop_size - n_pop_curr):
            filtered_pop.append(candidate_fronts[-1][sorted_final_front_id[i]])

        return filtered_pop

    def calculate_offspring_survival_rate(self, pop):
        n_child_survived = 0
        for p in pop:
            if p.pt == 0:
                n_child_survived += 1

        return n_child_survived/len(pop)

    def visualize_network(pop):
        output_dir = "network_plots"
        fig_format = "png"
        n_col = 6
        n_row = int(np.ceil(len(pop) / n_col))
        plt.figure(figsize=(5 * n_col, n_row * 20))
        for i in range(len(pop)):
            fig_path = (os.path.join(output_dir, "network_{}").format(pop[i].id))
            # create the network figure and save to file
            viz = make_dot_genome(pop[i].genome, format=fig_format)
            viz.render(fig_path, view=False)
            img = mpimg.imread(fig_path + "." + fig_format)
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

        return

def demo():
    args = nsga2_config.parse_args()
    nsga2 = Engine(args)
    pop, config_archive, conn_archive = nsga2.initialization()
    pop = nsga2.selection(pop)
    pop = nsga2.mutation(pop)
    for p in pop:
        p.display_content()

if __name__ == "__main__":
    demo()
