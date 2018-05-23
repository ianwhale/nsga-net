# -*- coding: utf-8 -*-
"""
Created on Sun Sep 03 17:11:13 2017

@author: Dhebar
modified by: Zhichao Lu 02-18-2018
"""

#..nsga-ii classes...
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plugins.genome_visualizer import make_dot_phase
from plugins.genome_visualizer import make_dot_genome

class Individual:
    def __init__(self):
        self.xreal = np.array([])
        self.fitness = np.array([0.0, 0.0])
        self.crowding_dist = 0.0
        self.rank = -1
        self.S_dom = []
        self.n_dom = -1
        self.address = -1

class Phase():
    def __init__(self, bit_string=[]):
        self.bit_string = bit_string
        self.id = -1
        self.equivalent_phases = []
        if len(self.bit_string) > 0:
            self.genome = self.phase_encoder(self.bit_string)
            self.key = self.phase_key_encoder(self.genome)
            self.active_nodes = self.count_phase_active_nodes(self.genome)
            self.connections = self.count_phase_connections(self.genome)

    def phase_encoder(self, phase_bit_string):
        n = int(np.sqrt(2 * len(phase_bit_string) - 7/4) - 1/2)
        genome = []
        for i in range(n):
            operator = []
            for j in range(i + 1):
                operator.append(phase_bit_string[int(i * (i + 1) / 2 + j)])
            genome.append(operator)
        genome.append([phase_bit_string[-1]])
        return genome

    def phase_key_encoder(self, genome):
        genome_key = []
        for op in genome:
            genome_key.append("".join(map(str, op)))
        return '-'.join(genome_key)

    @staticmethod
    def count_phase_active_nodes(genome):
        active_nodes = len(genome)
        for i in range(len(genome) - 1):
            if np.sum(genome[i]) < 1:
                non_active_node = 1
                for j in range(i + 1, len(genome) - 1):
                    if genome[j][i + 1] > 0:
                        non_active_node = 0
                        break
                active_nodes = active_nodes - non_active_node

        first_node_redundant = 1
        for k in range(len(genome) - 1):
            if genome[k][0] > 0:
                first_node_redundant = 0
                break
        return active_nodes - first_node_redundant

    @staticmethod
    def count_phase_connections(genome):
        connections = sum([sum(node) for node in genome])
        # calculate output connection for each node
        output_connection = [0] * len(genome)
        for bit in range(len(genome) - 1):
            for node in range(bit, len(genome) - 1):
                if genome[node][bit] > 0:
                    output_connection[bit] = 1
                    break
        # calculate input connection for each node
        input_connection = [0]
        for node in range(len(genome) - 1):
            if sum(genome[node]) > 0:
                input_connection.append(1)
            else:
                input_connection.append(0)
        for bit in range(len(output_connection)):
            if output_connection[bit] != input_connection[bit]:
                connections += 1

        return connections

    def append_equivalent_phase(self, equivalent_phase):
        self.equivalent_phases.append(equivalent_phase)
        return

    def visualize_phase(self, show=False):
        output_dir = "network_plots"
        fig_format = "png"
        fig_path = (os.path.join(output_dir, "phase_{}").format(self.id))
        viz = make_dot_phase(self.genome, format=fig_format)
        viz.render(fig_path, view=False)
        if show:
            plt.figure(figsize=(5, 10))
            img = mpimg.imread(fig_path + "." + fig_format)
            plt.subplot(1, 1, 1)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    def visualize_equivalents(self, show=False):
        if len(self.equivalent_phases) > 0:
            output_dir = "network_plots"
            fig_format = "png"
            n_equivalent_phases = len(self.equivalent_phases)
            if show:
                plt.figure(figsize=(5*n_equivalent_phases, 10))
                for i in range(n_equivalent_phases):
                    fig_path = (os.path.join(output_dir, "phase_{}_{}").format(self.id, i))
                    viz = make_dot_phase(self.equivalent_phases[i].genome, format=fig_format)
                    viz.render(fig_path, view=False)
                    img = mpimg.imread(fig_path + "." + fig_format)
                    plt.subplot(1, n_equivalent_phases, i+1)
                    plt.imshow(img)
                    plt.axis('off')
                plt.show()

class IndividualEDNN(Individual):
    def __init__(self, bit_string=[], n_phases=3, n_nodes=4):
        Individual.__init__(self)
        self.bit_string = bit_string
        self.num_nodes_per_phase = n_nodes
        self.num_phases = n_phases
        if len(self.bit_string) > 0:
            self.phases, self.genome, self.key, self.active_nodes = self.encoder()
        self.phase_connection = []
        self.n_params = []
        self.n_FLOPs = []
        self.cr = 1  # 1 means from genetic operation, 0 means from derived heuristic
        self.pt = 0  # 1 means from parent pop pt, 0 means from child pop qt
        self.parents = []
        self.mut_option = 0  # 0 means not mutated, 1 means binary mutated, 2 means shuffle
        self.id = []  # this is the unique ID for each individual created

    def encoder(self):
        # start decoding for each phase
        phases = []
        genome = []
        key = []
        active_nodes = []
        for p in range(self.num_phases):
            phase = Phase(self.bit_string[p])
            phases.append(phase)
            genome.append(phase.genome)
            key.append(phase.key)
            active_nodes.append(phase.active_nodes)

        return phases, genome, key, active_nodes

    def display_content(self):
        # print(self.bit_string)
        print("{}: ".format(self.key))
        print("Mutation applied = {}".format(self.mut_option))
        print("Number of active nodes = {}".format(sum(self.active_nodes)))
        print("Classification error = {:.3f}, number of params = {:.3f}".format(
            self.fitness[0], self.n_params))
        # if len(self.equivalent_keys) > 0:
        #     print("Equivalent found so far includes: ")
        #     for i in range(len(self.equivalent_keys)):
        #         print("{}: {}".format(i, self.equivalent_keys[i]))

    def render_networks(self, fig_path, fig_format):
        """
        Renders the graphviz and image files of network architecture defined by a genome.
        :param population: list of nsga individuals.
        :param nsga_details: bool, true if we want rank and crowding distance in the title.
        :param show_genome: bool, true if we want the genome in the title.
        """

        viz = make_dot_genome(self.genome, format=fig_format)
        viz.render(fig_path, view=False)

    def visualize_network(self, show=False):
        output_dir = "network_plots"
        fig_format = "png"
        fig_path = (os.path.join(output_dir, "indv_{}").format(self.id))

        # create the network figure and save to file
        self.render_networks(fig_path, fig_format)

        if show:
            plt.figure(figsize=(5, 10))
            img = mpimg.imread(fig_path + "." + fig_format)
            plt.subplot(1, 1, 1)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

def demo():
    indv = IndividualEDNN([[1]*7, [1]*7, [1]*7])
    indv.equivalent_keys.append(indv.key)
    indv.display_content()
    # indv.visualize_network()

if __name__ == "__main__":
    demo()