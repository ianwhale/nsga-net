# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 14:40:18 2017

@author: Zhichao

the selection is based on the 'dominance' check. Rank information is 
considered obsolete here
"""
# NSGA2 main script
# import config
import numpy as np
import nsga2_config
import nsga2_engine

def Main():
    args = nsga2_config.parse_args()
    # args_pytorchnet = config.parser.parse_args()

    run = args.run  # current run id
    np.random.seed(args.seed)

    print('run = {}'.format(run))

    # initiate the nsga2 engine
    nsga2 = nsga2_engine.Engine(args, run)

    # write configuration to file
    nsga2.report_config_to_file(args)

    # ============================ INITIALIZATION =============================== #
    # if resume from previous generation
    if nsga2.resume:
        pop, config_archive, pop_archive = nsga2.resume_from_previous()
        init_gen = nsga2.resume_gen
        print("Resume from generation {}".format(nsga2.resume_gen))

    # if not, start the normal initialization process
    else:
        # initialization
        pop, config_archive = nsga2.initialization()
        init_gen = 0
        # evaluate initial population
        nsga2.assign_fitness(pop, args)
        nsga2.assign_rank_crowding_distance(pop)
        pop_archive = nsga2.update_pop_archive(pop, [])
        # record initial statistics
        pop = nsga2.report_gen_to_file(run, 0, pop, [], config_archive, pop_archive)
        print("Initialization completed")

    # # ============================ EXPLORATION =============================== #
    # exploration stopping criteria related constant
    alpha = []  # number of offspring survived to next generation
    print("Exploration starts")
    for gen in range(init_gen+1, args.max_gen_exploration):
        print('gen = {}'.format(gen))
        child_pop = nsga2.selection(pop)
        child_pop = nsga2.mutation(child_pop, gen)
        child_pop, config_archive = nsga2.prepare_pop_for_evaluation(gen, child_pop,
                                                                     config_archive, pop_archive)
        child_pop, pop_archive = nsga2.compute_fitness_pop(child_pop, pop_archive,
                                                           args)
        mixed_pop = pop + child_pop
        pop = nsga2.fill_nondominated_sort(mixed_pop)
        # calculate offspring survival rate
        offspring_survival_rate = nsga2.calculate_offspring_survival_rate(pop)
        pop = nsga2.report_gen_to_file(run, gen, pop, child_pop, config_archive, pop_archive)
        # early termination check
        alpha.append(offspring_survival_rate)
        if (np.mean(alpha[-5:]) < args.term_threshold) and args.term:
            break

    # ============================ EXPLOITATION =============================== #
    # # ad-hoc
    # gen = init_gen
    # reduce the pop size by half
    # nsga2.pop_size = int(nsga2.pop_size/2)
    # # modified the pytorchnet configuration for moderate training
    # args_pytorchnet.nepochs = int(args_pytorchnet.nepochs*2)
    #
    # # re-evaluate the current parent_pop
    # print("Preparation for Exploitation")
    # nsga2.assign_fitness(pop, args_pytorchnet)
    # # update the pop archive with the new fitness
    # pop_archive = nsga2.update_pop_archive(pop, pop_archive, True)
    if 'gen' in locals():
        pass
    else:
        gen = init_gen
    print("Exploitation starts")
    for gen in range(gen+1, args.max_gen_exploration+args.max_gen_exploitation):
        print('gen = {}'.format(gen))
        child_pop, config_archive = nsga2.prepare_pop_for_evaluation(gen, [], config_archive,
                                                                     pop_archive, True)
        child_pop, pop_archive = nsga2.compute_fitness_pop(child_pop, pop_archive,
                                                           args)
        mixed_pop = pop + child_pop
        pop = nsga2.fill_nondominated_sort(mixed_pop)
        pop = nsga2.report_gen_to_file(run, gen, pop, child_pop, config_archive, pop_archive)

    # ============================ EXTRAPOLATION =============================== #
    # print("Extrapolation starts")
    # pop_final = []  # only non-dominated solutions found so far will be extrapolated
    # for p in pop:
    #     if p.rank == 1:
    #         pop_final.append(p)
    # # modified the pytorchnet configuration for throughout training
    # args_pytorchnet.nfilters, args_pytorchnet.nepochs = 16, 30
    # args_pytorchnet.optim_method = 'SGD'
    # args_pytorchnet.learning_rate, args_pytorchnet.min_learning_rate = 0.1, 1e-6
    # nsga2.assign_fitness(pop_final, args_pytorchnet)
    # nsga2.assign_rank_crowding_distance(pop_final)
    # nsga2.report_gen_to_file(run, args.max_gen_exploration +
    #                          args.max_gen_exploitation, pop_final, [],
    #                          config_archive, conn_archive)
    return

if __name__ == "__main__":
    Main()