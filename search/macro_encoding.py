# similar encoding to Genetic CNN paper
# we add one more skip connection bit
# L. Xie and A. Yuille, "Genetic CNN,"
# 2017 IEEE International Conference on Computer Vision (ICCV)
import numpy as np


def phase_dencode(phase_bit_string):
    n = int(np.sqrt(2 * len(phase_bit_string) - 7/4) - 1/2)
    genome = []
    for i in range(n):
        operator = []
        for j in range(i + 1):
            operator.append(phase_bit_string[int(i * (i + 1) / 2 + j)])
        genome.append(operator)
    genome.append([phase_bit_string[-1]])
    return genome


def convert(bit_string, n_phases=3):
    # assumes bit_string is a np array
    assert bit_string.shape[0] % n_phases == 0
    phase_length = bit_string.shape[0] // n_phases
    genome = []
    for i in range(0, bit_string.shape[0], phase_length):
        genome.append((bit_string[i:i+phase_length]).tolist())

    return genome


def decode(genome):
    genotype = []
    for gene in genome:
        genotype.append(phase_dencode(gene))

    return genotype


if __name__ == "__main__":
    n_phases = 3
    # bit_string = np.random.randint(0, 2, size=21)
    # print(bit_string)
    # genome = decode(convert(bit_string, n_phases))
    # print(genome)
    #
    # channels = [(3, 128), (128, 128), (128, 128)]
    #
    # out_features = 10
    #
    # import torch
    # from models.macro_models import EvoNetwork
    # from misc import utils
    #
    # data = torch.randn(1, 3, 32, 32)
    # net = EvoNetwork(genome, channels, out_features, (32, 32), decoder='dense')
    # print("param size = {}MB".format(utils.count_parameters_in_MB(net)))
    # output = net(torch.autograd.Variable(data))
    #
    # print(output)