# model_utils.py

from torch.autograd import Variable


def prod(items):
    """
    Calculate the product of a list of numbers.
    :param items: list of numbers.
    :return: float|int
    """
    total = 1
    for item in items:
        total *= item

    return total


class ParamCounter:
    def __init__(self, output_var):
        assert isinstance(output_var, Variable), "Type {} is not a torch.autograd.Variable.".format(type(output_var))

        self.seen = set()
        self.count = 0

        self.recurse_backprop(output_var.grad_fn)

    def recurse_backprop(self, var):
        """
        Recurse through backprop graph to count parameters.
        :param var: torch.autograd.Variable
        """
        if var not in self.seen:
            if hasattr(var, "variable"):
                u = var.variable
                self.count += prod(u.size())

            self.seen.add(var)

            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        self.recurse_backprop(u[0])

            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    self.recurse_backprop(t)

    def get_count(self):
        """
        Get parameter count.
        :return:
        """
        return self.count


def demo():
    import torch
    import models
    """
    This demo uses EvoNetwork, however the output of any type of PyTorch network will work.
    """
    genome = [[
        [1],
        [0, 0],
        [1, 1, 0],
        [1]
    ]]

    channels = [(3, 8)]
    out_features = 10
    net = models.EvoNetwork(genome, channels, out_features, (32, 32))

    data = torch.randn(3, 3, 32, 32)
    output = net(Variable(data))

    print("Trainable Parameters: {}".format(ParamCounter(output).get_count()))


if __name__ == "__main__":
    demo()