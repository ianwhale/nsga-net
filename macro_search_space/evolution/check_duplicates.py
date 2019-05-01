import torch
import torch.nn as nn
import models
import math

def weights_init(m):
    torch.manual_seed(0)
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        # m.weight.data.fill_(0.32456)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(0.5) # this what to do
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.fill_(0.6)
        m.bias.data.zero_()

def get_torch_model_output(genome,seed):
    torch.manual_seed(seed)
    # One input channel, 8 output channels.
    channels = [(3, 8)]
    if len(genome) > 1:
        channels.append([8, 8])

    if len(genome) > 2:
        channels.append([8, 8])
    # channels = [(3, 8), (8, 8), (8, 8)]
    # print(channels)
    out_features = 1
    data = torch.randn(1, 3, 32, 32)
    net = models.evonetwork.EvoNetwork(genome, channels, out_features, (32, 32))
    net.apply(weights_init)
    return net(torch.autograd.Variable(data))

def check_torch_model(genome1, genome2):
    # function that builds the torch model based on two genome and pass a few arbitrary inputs
    # and try to match outputs to determine if two models are the same
    equivalent_flag = True  # assumes two genomes are the same to start with

    seeds = [100, 200, 300, 500, 600, 700, 800, 900]
    for seed in seeds:
        output1 = get_torch_model_output(genome1, seed)
        output2 = get_torch_model_output(genome2, seed)

        diff = output1 - output2
        # diff = output1.data.numpy() - output2.data.numpy()
        diff = diff.data.numpy()
        # print(diff)
        if all(abs(diff) > 1e-4):
            equivalent_flag = False
            break
    return equivalent_flag

def check_effective_genome(genome):
    effective_genome = []
    # loop through each stage
    for stage in genome:
        stage_effective = False  # assumes this stage is redundant
        # loop through each node in this stage, NOT including the last residual connection
        for i in range(len(stage)-1):
            # print(i)
            # print(all(v == 0 for v in stage[i]))
            for conn in stage[i]:
                if conn > 0:
                    stage_effective = True
                    break
            if stage_effective:
                break
        if stage_effective:
            effective_genome.append(stage)

    return effective_genome

def count_node_IO(genome):
    node_IO = dict()
    keys = []
    # first node:
    key = '0-{}'.format(sum(x[0] for x in genome[:-1]))
    keys.append(key)
    for i in range(1, len(genome)-1):
        key = '{}-{}'.format(sum(genome[i-1]), sum(x[i] for x in genome[:-1] if len(x) > i))
        keys.append(key)
    # last node
    key = '{}-0'.format(sum(genome[-2]))
    keys.append(key)
    unique_keys = list(set(keys))
    for k in unique_keys:
        node_IO[k] = sum(k == key for key in keys)

    return node_IO

def check_node_IO(genome1, genome2):
    equivalent = False
    node_IO1 = count_node_IO(genome1)
    node_IO2 = count_node_IO(genome2)
    if node_IO1.keys() == node_IO2.keys():
        for key in node_IO1:
            if node_IO1[key] != node_IO2[key]:
                break
        equivalent = True
    return equivalent

def overall_check(indv1, indv2):
    # 1st check, get the effective genome and check bit by bit
    eff_genome1 = check_effective_genome([indv1.genome])
    eff_genome2 = check_effective_genome([indv2.genome])
    # eff_genome1, eff_genome2 = genome1, genome2
    # print(eff_genome1)
    # print(eff_genome2)
    if eff_genome1 == eff_genome2:
        return True

    # 2nd check the number of active nodes
    if indv1.active_nodes != indv2.active_nodes:
        return False

    # 3rd check the node IO
    if not(check_node_IO(indv1.genome, indv2.genome)):
        return False

    # print(duplicates_flag)
    # 2nd check the pytorch model
    if not(check_torch_model(eff_genome1, eff_genome2)):
        return False

    return True

    
def demo ():
    genome1 = [[
        [1],  # A_2 connections.
        [0, 1],  # A_3 connections.
        [1, 0, 1],  # A_4 connections.
        [1]  # A_5 connections (do we connect to A_0?)
    ], [
        [0],
        [1, 0],
        [1, 1, 1],
        [0]
    ], [
        [0],  # A_2 connections.
        [1, 0],  # A_3 connections.
        [0, 0, 0],  # A_4 connections.
        [0]  # A_5 connections (do we connect to A_0?)
    ]
    ]
    genome2 = [[
        [0],  # A_2 connections.
        [1, 0],  # A_3 connections.
        # [1, 1, 1],  # A_4 connections.
        [0]  # A_5 connections (do we connect to A_0?)
    ], [
        [0],
        [1, 0],
        # [1, 1, 1],
        [0]
    ], [
        [0],  # A_2 connections.
        [1, 0],  # A_3 connections.
        # [0, 0, 0],  # A_4 connections.
        [0]  # A_5 connections (do we connect to A_0?)
    ]
    ]

    print(overall_check(genome1, genome2))


if __name__ == "__main__":
    demo()