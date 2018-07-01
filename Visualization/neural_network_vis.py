""" Visualization Library for Neural Networks """
import matplotlib.pyplot as pyplot
import math

# TODO RED FOR DISABLED LINK


class VisNetwork:
    def __init__(self, neurons):
        self.neurons = neurons


class VisNeuron:
    """
    A node in a neural network which models the functions of a dendrite,
    by having the nodes through which the neural cell receives data.

    incoming_neurons:   List of the labels of the neural nodes that provide
                        information for this neuron
    weights:            List of the weights of the connections from other
                        neurons. The weight at index i corresponds to the
                        connection from the neuron at incoming_neurons index
                        i to the current neuron
    """

    def __init__(self, incoming_neurons, weights, incoming_enabled, label):
        self.incoming_neurons = incoming_neurons
        self.weights = weights
        self.incoming_enabled = incoming_enabled
        self.label = label


class NNVis:
    def __init__(self, network):
        self.network = network
        self.layers = find_layers(network)
        self.bounds = self.get_bounds()
        self.structure = {}
        self.connections = []

    def display_all(self):
        pyplot.show()

    def construct_structure(self):

        x_len, y_len = self.bounds
        pyplot.axis([-1, 2 * x_len - 1, -y_len, y_len])

        for layer_num in range(0, len(self.layers)):
            layer = self.layers[layer_num]
            size_boost = len(layer) - 1
            x = layer_num * 2
            for neuron_num in range(0, len(layer)):
                y = size_boost - (neuron_num * 2)
                neuron = layer[neuron_num]
                cur_circle = pyplot.Circle((x,y), .5, fill=False, label=neuron)
                self.structure[neuron] = cur_circle
                pyplot.text(x, y, str(neuron), ha="center", va="center", size=12)
                pyplot.gca().add_patch(cur_circle)

    def construct_connections(self, show_disabled=False):

        color_options = ["r-", "b-", "y-", "m-", "g-", "c-", "k-"]

        for neuron in self.network.neurons:
            circle1 = self.structure[neuron.label]

            for conn_index in range(0, len(neuron.incoming_neurons)):
                conn = neuron.incoming_neurons[conn_index]
                is_enabled = neuron.incoming_enabled[conn_index]

                if show_disabled | is_enabled:
                    circle2 = self.structure[conn]
                    (x1, y1), (x2, y2) = self.find_connection_line(circle1, circle2)
                    layer_span = int(abs(circle1.center[0] - circle2.center[0]) / 2)
                    if layer_span > len(color_options):
                        color = "k-"
                    else:
                        color = color_options[layer_span - 1]
                    pyplot.plot([x1, x2], [y1, y2], color, 2)

    # Method which finds the appropriate line length between two circles so as not to intersect inside either circle
    def find_connection_line(self, circle1, circle2):
        """


        :param circle1:     The first matplotlib.pyplot.Circle
        :param circle2:     The second matplotlib.pyplot.Circle
        :return:            The adjusted line that does not intersect either circle
        """

        if circle2.center[1] > circle1.center[1]:
            temp = circle1
            circle1 = circle2
            circle2 = temp

        circle1x, circle1y = circle1.center
        circle2x, circle2y = circle2.center

        if circle2y - circle1y == 0:
            return (circle1x - .5, circle1y), (circle2x + .5, circle2y)

        x, y = (circle1x - circle2x, circle1y - circle2y)
        mag = math.sqrt(math.pow((circle2x - circle1x), 2) + math.pow((circle2y - circle1y), 2))
        adj_x = x / (mag * 2)
        adj_y = y / (mag * 2)

        return (circle1x - adj_x, circle1y - adj_y), (circle2x + adj_x, circle2y + adj_y)

    # TODO Implement showing various neuron values for given input
    def display_example(self, input):
        raise NotImplementedError("Method not implemented yet")

    def get_bounds(self):

        layers = self.layers
        x_length = len(layers)
        y_length = 0

        for layer in layers:
            if len(layer) > y_length:
                y_length = len(layer)

        return x_length, y_length

# TODO Create function to find layers for recurrent networks (this only works for feed forward)
def find_layers(network):
    """
    A topological sort (using Kahn's Algorithm) that identifies the
    different layers that are implicit in the given neural network

    :param network:   A Network
    :return:        A list representing the topological order
                    of the nodes in the neural network
    """

    copy_neurons = copy_network(network).neurons

    labels = []
    is_used = []
    for neuron in copy_neurons:
        labels.append(neuron.label)
        is_used.append(False)

    labels = sorted(labels)
    goal_size = len(labels)

    adj_list = [[] for _ in range(0, len(labels))]

    # This adjacency list is atypical in that at each index i
    # there is a list of indices such that each element j
    # in that list denotes an edge (j,i) in the graph.
    # (As opposed to the typical (i,j) representation)
    adj_list_rev = [[] for _ in range(0, len(labels))]

    layer_list = []
    num_sorted = 0
    next_list = []

    # Fills the two adjacency lists with the edges from the genes
    for neuron in copy_neurons:
        into = labels.index(neuron.label)
        for inc_neuron in neuron.incoming_neurons:
            out = labels.index(inc_neuron)
            adj_list_rev[into].append(out)
            adj_list[out].append(into)

    while not (goal_size <= num_sorted):

        cur_list = []

        # Creates the initial list from the nodes with no incoming edges
        # There is required to be at least one or the neural network is invalid
        for node_num in range(0, len(labels)):
            if (not adj_list_rev[node_num]) & (not is_used[node_num]):
                cur_list.append(node_num)
                is_used[node_num] = True

        num_sorted += len(cur_list)
        layer_labels = []
        # Converts cur_list to the list of labels representing the neurons
        for item in cur_list:
            layer_labels.append(labels[item])
        layer_list.append(layer_labels)

        # For each neuron in the current layer remove outgoing and incoming edges
        for cur_node in cur_list:
            while adj_list[cur_node]:
                cur_edge = adj_list[cur_node].pop(0)
                adj_list_rev[cur_edge].remove(cur_node)
                if not adj_list_rev[cur_edge]:
                    next_list.append(cur_edge)

    return layer_list


def copy_network(network):

    neurons = []
    for neuron in network.neurons:
        neurons.append(copy_neuron(neuron))
    new_network = VisNetwork(neurons)
    return new_network


def copy_neuron(neuron):

    return VisNeuron(neuron.incoming_neurons, neuron.weights, neuron.incoming_enabled, neuron.label)