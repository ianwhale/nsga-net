# NASNet Search Space https://arxiv.org/pdf/1707.07012.pdf
# code modified from DARTS https://github.com/quark0/darts
import numpy as np
from collections import namedtuple

import torch
from models.micro_models import NetworkCIFAR as Network

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_norm = namedtuple('Genotype', 'normal normal_concat')
Genotype_redu = namedtuple('Genotype', 'reduce reduce_concat')

# what you want to search should be defined here and in micro_operations
PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'sep_conv_7x7',
    'conv_7x1_1x7',
]


def convert_cell(cell_bit_string):
    # convert cell bit-string to genome
    tmp = [cell_bit_string[i:i + 2] for i in range(0, len(cell_bit_string), 2)]
    return [tmp[i:i + 2] for i in range(0, len(tmp), 2)]


def convert(bit_string):
    # convert network bit-string (norm_cell + redu_cell) to genome
    norm_gene = convert_cell(bit_string[:len(bit_string)//2])
    redu_gene = convert_cell(bit_string[len(bit_string)//2:])
    return [norm_gene, redu_gene]


def decode_cell(genome, norm=True):

    cell, cell_concat = [], list(range(2, len(genome)+2))
    for block in genome:
        for unit in block:
            cell.append((PRIMITIVES[unit[0]], unit[1]))
            if unit[1] in cell_concat:
                cell_concat.remove(unit[1])

    if norm:
        return Genotype_norm(normal=cell, normal_concat=cell_concat)
    else:
        return Genotype_redu(reduce=cell, reduce_concat=cell_concat)


def decode(genome):
    # decodes genome to architecture
    normal_cell = genome[0]
    reduce_cell = genome[1]

    normal, normal_concat = [], list(range(2, len(normal_cell)+2))
    reduce, reduce_concat = [], list(range(2, len(reduce_cell)+2))

    for block in normal_cell:
        for unit in block:
            normal.append((PRIMITIVES[unit[0]], unit[1]))
            if unit[1] in normal_concat:
                normal_concat.remove(unit[1])

    for block in reduce_cell:
        for unit in block:
            reduce.append((PRIMITIVES[unit[0]], unit[1]))
            if unit[1] in reduce_concat:
                reduce_concat.remove(unit[1])

    return Genotype(
        normal=normal, normal_concat=normal_concat,
        reduce=reduce, reduce_concat=reduce_concat
    )


def compare_cell(cell_string1, cell_string2):
    cell_genome1 = convert_cell(cell_string1)
    cell_genome2 = convert_cell(cell_string2)
    cell1, cell2 = cell_genome1[:], cell_genome2[:]

    for block1 in cell1:
        for block2 in cell2:
            if block1 == block2 or block1 == block2[::-1]:
                cell2.remove(block2)
                break
    if len(cell2) > 0:
        return False
    else:
        return True


def compare(string1, string2):

    if compare_cell(string1[:len(string1)//2],
                    string2[:len(string2)//2]):
        if compare_cell(string1[len(string1)//2:],
                        string2[len(string2)//2:]):
            return True

    return False


def debug():
    # design to debug the encoding scheme
    seed = 0
    np.random.seed(seed)
    budget = 2000
    B, n_ops, n_cell = 5, 7, 2
    networks = []
    design_id = 1
    while len(networks) < budget:
        bit_string = []
        for c in range(n_cell):
            for b in range(B):
                bit_string += [np.random.randint(n_ops),
                               np.random.randint(b + 2),
                               np.random.randint(n_ops),
                               np.random.randint(b + 2)
                               ]

        genome = convert(bit_string)
        # check against evaluated networks in case of duplicates
        doTrain = True
        for network in networks:
            if compare(genome, network):
                doTrain = False
                break

        if doTrain:
            genotype = decode(genome)
            model = Network(16, 10, 8, False, genotype)
            model.drop_path_prob = 0.0
            data = torch.randn(1, 3, 32, 32)
            output, output_aux = model(torch.autograd.Variable(data))
            networks.append(genome)
            design_id += 1
            print(design_id)


if __name__ == "__main__":
    # debug()
    # genome1 = [[[[3, 0], [3, 1]], [[3, 0], [3, 1]],
    #             [[3, 1], [2, 0]], [[2, 0], [5, 2]]],
    #            [[[0, 0], [0, 1]], [[2, 2], [0, 1]],
    #             [[0, 0], [2, 2]], [[2, 2], [0, 1]]]]
    # genome2 = [[[[3, 1], [3, 0]], [[3, 1], [3, 0]],
    #             [[3, 1], [2, 0]], [[2, 0], [5, 2]]],
    #            [[[0, 1], [0, 0]], [[2, 2], [0, 1]],
    #             [[0, 0], [2, 2]], [[2, 2], [0, 0]]]]
    #
    # print(compare(genome1, genome2))
    # print(genome1)
    # print(genome2)
    # bit_string1 = [3,1,3,0,3,1,3,0,3,1,2,0,2,0,5,2,0,0,0,1,2,2,0,1,0,0,2,2,2,2,0,1]
    # bit_string2 = [3, 0, 3, 1, 3, 0, 3, 1, 3, 1, 2, 0, 2, 0, 5, 2,
    #                0, 0, 0, 1, 2, 2, 0, 1, 0, 0, 2, 2, 2, 2, 0, 1]
    # # print(convert(bit_string1))
    # print(compare(bit_string1, bit_string2))
    # print(decode(convert(bit_string)))

    cell_bit_string = [3, 0, 3, 1, 3, 0, 3, 1, 3, 1, 2, 0, 2, 0, 5, 2]
    print(decode_cell(convert_cell(cell_bit_string), norm=False))
