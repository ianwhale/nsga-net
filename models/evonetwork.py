# evonetwork.py

import torch
import torch.nn as nn
from evolution import ResidualGenomeDecoder, VariableGenomeDecoder, DenseGenomeDecoder


def get_decoder(decoder_str, genome, channels):
    """
    Construct the appropriate decoder.
    :param decoder_str: string, refers to what genome scheme we're using.
    :param genome: list, list of genomes.
    :param channels: list, list of channel sizes.
    :return: evolution.ChannelBasedDecoder
    """
    if decoder_str == "residual":
        return ResidualGenomeDecoder(genome, channels)

    if decoder_str == "swapped-residual":
        return ResidualGenomeDecoder(genome, channels, preact=True)

    if decoder_str == "dense":
        return DenseGenomeDecoder(genome, channels)

    if decoder_str == "variable":
        return VariableGenomeDecoder(genome, channels)

    raise NotImplementedError("Decoder {} not implemented.".format(decoder_str))


class EvoNetwork(nn.Module):
    """
    Entire network.
    Made up of Phases.
    """
    def __init__(self, genome, channels, out_features, data_shape, decoder="residual"):
        """
        Network constructor.
        :param genome: depends on decoder scheme, for most this is a list.
        :param channels: list of desired channel tuples.
        :param out_features: number of output features.
        :param decoder: string, what kind of decoding scheme to use.
        """
        super(EvoNetwork, self).__init__()

        assert len(channels) == len(genome), "Need to supply as many channel tuples as genes."

        self.model = get_decoder(decoder, genome, channels).get_model()

        #
        # After the evolved part of the network, we would like to do global average pooling and a linear layer.
        # However, we don't know the output size so we do some forward passes and observe the output sizes.
        #

        out = self.model(torch.autograd.Variable(torch.zeros(1, channels[0][0], *data_shape)))
        shape = out.data.shape

        self.gap = nn.AvgPool2d(kernel_size=(shape[-2], shape[-1]), stride=1)

        shape = self.gap(out).data.shape

        self.linear = nn.Linear(shape[1] * shape[2] * shape[3], out_features)

        # We accumulated some unwanted gradient information data with those forward passes.
        self.model.zero_grad()

    def forward(self, x):
        """
        Forward propagation.
        :param x: Variable, input to network.
        :return: Variable.
        """
        x = self.gap(self.model(x))

        x = x.view(x.size(0), -1)

        return self.linear(x)


def demo():
    """
    Demo creating a network.
    """
    # Genome should be a list of genes describing phase connection schemes.
    genome = [
        [
            [1],
            [0, 0],
            [1, 1, 1],
            [1]
        ],
        [  # Phase will be ignored, there are no active connections (residual is not counted as active).
            [0],
            [0, 0],
            [0, 0, 0],
            [1]
        ],
        [
            [1],
            [0, 0],
            [1, 1, 1],
            [0, 0, 1, 0],
            [1]
        ]
    ]

    channels = [(3, 8), (8, 8), (8, 8)]

    out_features = 10
    data = torch.randn(16, 3, 32, 32)
    net = EvoNetwork(genome, channels, out_features, (32, 32))

    output = net(torch.autograd.Variable(data))

    print(output)


if __name__ == "__main__":
    demo()
