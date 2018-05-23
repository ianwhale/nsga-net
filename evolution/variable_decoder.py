# variable_decoder.py

import torch
import torch.nn as nn
from evolution.decoder import Decoder, Identity
from evolution.residual_decoder import phase_active, ResidualPhase


class VariableGenomeDecoder(Decoder):
    """
    Residual decoding with extra integer for type of node inside the phase.
    This genome decoder produces networks that are a superset of ResidualGenomeDecoder networks.
    """
    RESIDUAL = 0
    PREACT_RESIDUAL = 1
    DENSE = 2

    def __init__(self, list_genome, channels):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network, and the type of phase.
        :param channels: list, list of tuples describing the channel size changes.
        """
        super().__init__(list_genome)

        self._model = None

        phase_types = [gene.pop() for gene in list_genome]

        self._genome, self._types = self.get_effective_phases(list_genome, phase_types)
        self._channels = channels[:len(self._genome)]

        if not self._genome:
            self._model = Identity()
            return

        phases = []
        for idx, (gene, (in_channels, out_channels), phase_type) in enumerate(zip(self._genome,
                                                                                  self._channels,
                                                                                  self._types)):
            if phase_type == self.RESIDUAL:
                phases.append(ResidualPhase(gene, in_channels, out_channels, idx))

            elif phase_type == self.PREACT_RESIDUAL:
                phases.append(ResidualPhase(gene, in_channels, out_channels, idx, preact=True))

            elif phase_type == self.DENSE:
                phases.append(DensePhase(gene, in_channels, out_channels, idx))

            else:
                raise NotImplementedError("Phase type corresponding to {} not implemented.".format(phase_type))

        self._model = nn.Sequential(*self.build_layers(phases))

    @staticmethod
    def get_effective_phases(genome, phase_types):
        """
        Get only the phases that are active.
        Similar to ResidualDecoder.get_effective_genome but we need to consider phases too.
        :param genome:
        :param phase_types:
        :return:
        """
        effecive_genome = []
        effective_types = []

        for gene, phase_type in zip(genome, phase_types):
            if phase_active(gene):
                effecive_genome.append(gene)
                effective_types.append(*phase_type)

        return effecive_genome, effective_types

    def get_model(self):
        return self._model


class DensePhase(nn.Module):
    """
    Phase with nodes that operates like DenseNet's bottle necking and growth rate scheme.
    Refer to: https://arxiv.org/pdf/1608.06993.pdf
    """
    def __init__(self, gene, in_channels, out_channels, idx):
        """
        Constructor.
        :param gene: list, element of genome describing connections in this phase.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        :param idx: int, index in the network.
        """
        super(DensePhase, self).__init__()

        self.in_channel_flag = in_channels != out_channels  # Flag to tell us if we need to increase channel size.
        self.out_channel_flag = out_channels != DenseNode.t
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1 if idx != 0 else 3, stride=1, bias=False)
        self.dependency_graph = ResidualPhase.build_dependency_graph(gene)

        channel_adjustment = 0

        for dep in self.dependency_graph[len(gene) + 1]:
            if dep == 0:
                channel_adjustment += out_channels

            else:
                channel_adjustment += DenseNode.t

        self.last_conv = nn.Conv2d(channel_adjustment, out_channels, kernel_size=1, stride=1, bias=False)

        nodes = []
        for i in range(len(gene)):
            if len(self.dependency_graph[i + 1]) > 0:
                channels = self.compute_channels(self.dependency_graph[i + 1], out_channels)
                nodes.append(DenseNode(channels))

            else:
                nodes.append(None)

        self.nodes = nn.ModuleList(nodes)
        self.out = nn.Sequential(
            self.last_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def compute_channels(dependency, out_channels):
        """
        Compute the number of channels incoming to a node.
        :param dependency: list, nodes that a particular node gets input from.
        :param out_channels: int, desired number of output channels from the phase.
        :return: int
        """
        channels = 0
        for d in dependency:
            if d == 0:
                channels += out_channels

            else:
                channels += DenseNode.t

        return channels

    def forward(self, x):
        if self.in_channel_flag:
            x = self.first_conv(x)

        outputs = [x]

        for i in range(1, len(self.nodes) + 1):
            if not self.dependency_graph[i]:  # Empty dependencies, no output to give.
                outputs.append(None)

            else:
                # Call the node on the depthwise concatenation of its inputs.
                outputs.append(self.nodes[i - 1](torch.cat([outputs[j] for j in self.dependency_graph[i]], dim=1)))

        if self.out_channel_flag and 0 in self.dependency_graph[len(self.nodes) + 1]:
            # Get the last nodes in the phase and change their channels to match the desired output.
            non_zero_dep = [dep for dep in self.dependency_graph[len(self.nodes) + 1] if dep != 0]

            return self.out(torch.cat([outputs[i] for i in non_zero_dep] + [outputs[0]], dim=1))

        if self.out_channel_flag:
            # Same as above, we just don't worry about the 0th node.
            return self.out(torch.cat([outputs[i] for i in self.dependency_graph[len(self.nodes) + 1]], dim=1))

        return self.out(torch.cat([outputs[i] for i in self.dependency_graph[len(self.nodes) + 1]]))


class DenseNode(nn.Module):
    """
    Node that operates like DenseNet layers.
    Refer to: https://arxiv.org/pdf/1608.06993.pdf
    """
    t = 32  # Growth rate fixed at 32 (a hyperparameter, although fixed in paper)
    k = 4   # Growth rate multiplier fixed at 4 (not a hyperparameter, this is from the definition of the dense layer).

    def __init__(self, in_channels):
        """
        Constructor.
        Only needs number of input channels, everything else is automatic from growth rate and DenseNet specs.
        :param in_channels: int, input channels.
        """
        super(DenseNode, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, self.t * self.k, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.t * self.k),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.t * self.k, self.t, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return self.model(x)


def demo():
    from plugins.backprop_visualizer import make_dot_backprop
    genome = [
        [
            [1],
            [0, 0],
            [1, 1, 1],
            [1],
            [2]
        ],
        [
            [0],
            [0, 0],
            [0, 0, 0],
            [1],
            [2]
        ],
        [
            [1],
            [0, 0],
            [1, 0, 1],
            [1, 1, 0, 1],
            [0],
            [2]
        ]
    ]

    channels = [(3, 8), (8, 16), (16, 32)]
    data = torch.randn(16, 3, 32, 32)

    model = VariableGenomeDecoder(genome, channels).get_model()
    out = model(torch.autograd.Variable(data))

    make_dot_backprop(out).view()


if __name__ == "__main__":
    demo()