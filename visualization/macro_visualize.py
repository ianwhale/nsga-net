# evo_visualizer.py
import sys
# sys.path.insert(0, '/path/to/nsga-net')
sys.path.insert(0, '/Users/zhichao.lu/Dropbox/2019/github/nsga-net')

from graphviz import Digraph
from models.macro_decoder import ResidualPhase
from models import macro_genotypes as genotypes


def get_graph_function(type):
    """
    Get the appropriate function to build a dependency graph.
    :param type:
    :return: callable
    """
    if type == "residual":
        return ResidualPhase.build_dependency_graph

    raise NotImplementedError("Genome type {} not supported.".format(type))


def make_dot_phase(phase, rankdir="UD", format="pdf", title=None, filename="genome", type="residual"):
    """
    Visualize a single phase.
    :param genome: list of lists.
    :param phase: int, index of phase to visualize.
    :param rankdir: direction graph is oriented "UD"=Vertical, "LR"=horizontal.
    :param format: output file format, jpg, png, etc.
    :param title: title of graph.
    :param filename: filename of graph.
    :param type: string, what kind of decoder should we use.
    :return: graphviz dot object.
    :return:
    """
    node_color = "lightblue"
    conv1x1_color = "white"
    sum_color = "green4"
    pool_color = "orange"
    phase_background_color = "gray"
    fc_color = "gray"
    node_shape = "circle"
    conv1x1_shape = "doublecircle"

    gene = phase
    graph_function = get_graph_function(type)
    graph = graph_function(gene)

    nodes = [("node_0", ' ')] + [("node_" + str(j + 1), ' ') for j in range(len(gene) + 1)]
    edges = []

    for sink, dependencies in graph.items():
        for source in dependencies:
            edges.append((nodes[source][0], nodes[sink][0]))

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(format=format, filename=filename+'.gv', node_attr=node_attr, graph_attr=dict(size="12,12"))
    dot.attr(rankdir=rankdir)

    if title:
        dot.attr(label=title+"\n\n")
        dot.attr(labelloc='t')

    dot.node(nodes[0][0], nodes[0][1], fillcolor=conv1x1_color, shape=conv1x1_shape)

    with dot.subgraph(name="cluster") as p:
        p.attr(fillcolor=phase_background_color, label='', fontcolor="black", style="filled")

        for i in range(1, len(nodes) - 1):
            if len(graph[i]) != 0:
                p.node(nodes[i][0], nodes[i][1], fillcolor=node_color, shape=node_shape)

    dot.node(nodes[-1][0], nodes[-1][1], fillcolor=sum_color, shape=node_shape)

    # Add edges.
    for edge in edges:
        dot.edge(*edge)

    dot.attr(dpi="300")

    return dot


def make_dot_genome(genome, rankdir="UD", format="pdf", title=None, filename="genome", type="residual"):
    """
    Graphviz representation of network created by genome.
    :param genome: list of lists.
    :param rankdir: direction graph is oriented "UD"=Vertical, "LR"=horizontal.
    :param format: output file format, jpg, png, etc.
    :param title: title of graph.
    :param filename: filename of graph.
    :param type: string, what kind of decoder should we use.
    :return: graphviz dot object.
    """
    node_color = "lightblue"
    conv1x1_color = "white"
    sum_color = "green4"
    pool_color = "orange"
    phase_background_color = "gray"
    fc_color = "gray"

    node_shape = "circle"
    conv1x1_shape = "doublecircle"

    structure = []

    # Build node ids and names to make building graph easier.
    for i, gene in enumerate(genome):
        all_zeros = sum([sum(t) for t in gene[:-1]]) == 0

        if all_zeros:
            continue  # Skip everything is a gene is all zeros.

        prefix = "gene_" + str(i)
        phase = ("cluster_" + str(i + 1), "Phase " + str(i + 1))

        nodes = [(prefix + "_node_0", ' ')] \
            + [(prefix + "_node_" + str(j + 1), ' ') for j in range(len(gene) + 1)]

        pool = (prefix + "_pool", "Pooling")


        edges = []
        graph_function = get_graph_function(type)
        graph = graph_function(gene)

        for sink, dependencies in graph.items():
            for source in dependencies:
                edges.append((nodes[source][0], nodes[sink][0]))

        structure.append(
            {
                "nodes": nodes,
                "edges": edges,
                "pool": pool,
                "phase": phase,
                "all_zeros": all_zeros,
                "graph": graph
            }
        )

    final_pool = structure[-1]["pool"]
    new_pool = (final_pool[0], "Avg. Pooling")
    structure[-1]["pool"] = new_pool

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(format=format, filename=filename+'.gv', node_attr=node_attr, graph_attr=dict(size="12,12"))
    dot.attr(rankdir=rankdir)

    if title:
        dot.attr(label=title+"\n\n")
        dot.attr(labelloc='t')

    dot.node(str("input"), "Input")

    for j, struct in enumerate(structure):
        nodes = struct['nodes']
        edges = struct['edges']
        phase = struct['phase']
        pool = struct['pool']
        graph = struct['graph']
        all_zeros = struct['all_zeros']

        # Add nodes.
        dot.node(nodes[0][0], nodes[0][1], fillcolor=conv1x1_color, shape=conv1x1_shape)

        if j > 0:
            dot.edge(structure[j - 1]['pool'][0], nodes[0][0])

        if not all_zeros:
            with dot.subgraph(name=phase[0]) as p:
                p.attr(fillcolor=phase_background_color, label='', fontcolor="black", style="filled")

                for i in range(1, len(nodes) - 1):
                    if len(graph[i]) != 0:
                        p.node(nodes[i][0], nodes[i][1], fillcolor=node_color, shape=node_shape)

        dot.node(nodes[-1][0], nodes[-1][1], fillcolor=sum_color, shape=node_shape)
        dot.node(*pool, fillcolor=pool_color)

        # Add edges.
        for edge in edges:
            dot.edge(*edge)

        dot.edge(nodes[-1][0], pool[0])

    dot.edge("input", structure[0]['nodes'][0][0])

    dot.node("linear", "Linear", fillcolor=fc_color)
    dot.edge(structure[-1]['pool'][0], "linear")

    # TODO: Add legend (?)

    return dot


def demo():
    """
    Demonstrate visualizing a genome.
    """
    genome = [[[0], [1, 0], [0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0, 1], [0]],
              [[0], [0, 0], [0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0, 0], [1]],
              [[0], [0, 1], [1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1, 0], [0]]]

    d = make_dot_genome(genome, title="Demo Genome", filename="test")
    d.view()

    # d = make_dot_phase(genome, 1)
    # d.view()


if __name__ == "__main__":
    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)
    d = make_dot_genome(genotype, title="{}".format(genotype_name), filename="macro_network")
    d.view()
