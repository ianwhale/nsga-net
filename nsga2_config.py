# config file for running nsga2 
import argparse


def parse_args():
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
    parser.add_argument('--eval-method', type=int, default=0,
                        help='0: non-cluster, 1: cluster, 2: dummy')

    # ======================= EDNN related Parameters =====================================
    parser.add_argument('--n-phases', type=int, default=3, help='number of phases')
    parser.add_argument('--n-nodes', type=int, default=6, help='number of nodes per phases')
    parser.add_argument('--bprob-time', type=int, default=1800,
                        help='estimated time for one back propagation training in seconds')

    args = parser.parse_args()
    return args
