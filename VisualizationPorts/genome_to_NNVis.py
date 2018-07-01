""" Converts Genomes into NNVis Objects """
import sys

# Adds PycharmProjects to the system path to allow importing other projects
sys.path.append('C:/Users/Andrew Butler/PycharmProjects')
from Visualization import neural_network_vis as nnvis
from MarketEval.learn_model import Genome

def genome_to_vis_network(genome):
    """
        Creates and returns the network that is
        modeled by 'genome'
        :param genome:      Any instantiation of Genome (from MarketEval.learn_model)
        :return:            A VisNetwork  (from Visualization.neural_network_vis)
    """

    labels = []
    for gene in genome.genes:
        if not labels.__contains__(gene.into):
            labels.append(gene.into)
        if not labels.__contains__(gene.out):
            labels.append(gene.out)

    labels = sorted(labels)

    neuron_list = []

    # Appends all the neurons to the neuron_list
    for x in range(0, len(labels)):
        neuron_list.append(nnvis.VisNeuron([], [], [], labels[x]))

    # Creates connections between neurons based off of genes in the genome
    for gene in genome.genes:
        cur_neuron = neuron_list[labels.index(gene.into)]
        cur_neuron.incoming_neurons.append(gene.out)
        cur_neuron.weights.append(gene.weight)
        cur_neuron.incoming_enabled.append(gene.enabled)

    return nnvis.VisNetwork(neuron_list)
