import random
import math

# connection gene that describes the weight between two neurons
class NeuronGene:
    def __init__(self, id, layer, bias = 0):
        self.id = id
        self.layer = layer
        self.value = 0
        self.bias = bias
        self.in_connections = []
        self.lookup_table = {}

    def add_connection(self, conn):
        self.in_connections.append(conn)

    # convert this into an iterative function
    def get_value(self):
        # squeezes numbers to be sharply between 0 and 1
        def sigmoid5(num):
            return 1/(1+math.e ** (-5 * num))
        valid_connections = list(filter(lambda x: x.expressed, self.in_connections))
        if len(valid_connections) == 0:
            return self.value
        self.value = 0
        for conn in valid_connections:
            self.value += conn.weight * conn.in_neuron.get_value()
        self.value += self.bias
        self.value = sigmoid5(self.value)
        return self.value

    def mutate_bias(self):
        self.bias += random.uniform(-1.5,1.5)

class ConnectionGene:
    innovation_number = 1
    @classmethod
    def get_innovation_number(cls):
        # class innovation number forces unique numbers for each connection gene
        return cls.innovation_number
    @classmethod
    def increment_innovation_number(cls):
        cls.innovation_number += 1

    def __init__(self, in_neuron, out_neuron, weight, innovation_number = None):
        self.in_neuron = in_neuron
        self.out_neuron = out_neuron
        self.weight = weight
        self.expressed = True
        if innovation_number is None:
            self.innovation_number = ConnectionGene.get_innovation_number()
        else:
            self.innovation_number = innovation_number
        ConnectionGene.increment_innovation_number()

    def mutate_weight(self):
        self.weight *= random.uniform(-1.5, 1.5)

    def randomize_weight(self):
        self.weight = random.uniform(-2, 2)

    def disable(self):
        self.expressed = False
