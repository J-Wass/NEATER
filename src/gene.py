import random
import math

class NeuronGene:
    global_id = 1
    @classmethod
    def get_global_id(cls):
        return cls.global_id
    @classmethod
    def increment_global_id(cls):
        cls.global_id += 1

    def __init__(self, id=None, layer=0, bias = 0, is_input = False, is_output = False, aggregation = 'sum'):
        self.layer = layer
        self.value = None
        self.bias = bias
        self.in_connections = []
        self.lookup_table = {}
        self.is_input = is_input
        self.is_output = is_output
        self.aggregation = aggregation
        if id is None:
            self.id = NeuronGene.get_global_id()
        else:
            self.id = id
        NeuronGene.increment_global_id()

    def add_connection(self, conn):
        self.in_connections.append(conn)

    def mutate_bias(self):
        # we don't add biases to input neurons around these parts
        if not self.is_input:
            self.bias += random.uniform(-0.5,0.5)

    def __repr__(self):
        ret_string = "Id{0}: \n\tvalue={5} \n\tlayer={1} \n\tbias={2} \n\tinput={3} \n\toutput={4}\n\n".format(self.id, self.layer,self.bias, self.is_input, self.is_output, self.value)
        return ret_string

class ConnectionGene:
    innovation_number = 1
    @classmethod
    def get_innovation_number(cls):
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

    def mutate_weight(self, randomization_chance = 0.10):
        self.weight *= random.uniform(0.5, 1.5)
        # on rare occasion, flip sign of weight
        if random.uniform(0,1) < randomization_chance:
            self.weight = random.uniform(-2,2)

    def disable(self):
        self.expressed = False

    def __repr__(self):
        ret_string = "Innovation{0}: \n\tin_neuron={1} \n\tout_neuron={1} \n\tweight={2} \n\texpressed={3} \n\n".format(self.innovation_number, self.in_neuron.id,self.out_neuron.id, self.weight, self.expressed)
        return ret_string
