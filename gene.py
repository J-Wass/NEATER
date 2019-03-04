import random

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

    # stack used to calculate the value of the output neurons
    def get_value(self):
        valid_connections = list(filter(lambda x: x.expressed, self.in_connections))
        node_stack = []
        activation = 0
        for conn in valid_connections:
            node_stack.append((conn, 1.0)) # tuples are (connection_gene, multiplier)
        while len(node_stack) > 0:
            current_node = node_stack.pop()
            connection_gene = current_node[0]
            multiplier = current_node[1]
            activation += connection_gene.in_neuron.bias * multiplier
            new_connections = list(filter(lambda x: x.expressed, conn.in_neuron.in_connections))
            if len(new_connections) == 0:
                activation += connection_gene.weight * connection_gene.in_neuron.value * multiplier
            else:
                for conn in new_connections:
                    val = connection_gene.weight * multiplier
                    node_stack.append((conn, connection_gene.weight * multiplier))
        return activation

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
