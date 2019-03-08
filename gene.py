import random
import math

# connection gene that describes the weight between two neurons
class NeuronGene:
    def __init__(self, id, layer, bias = 0, is_input = False):
        self.id = id
        self.layer = layer
        self.value = None
        self.bias = bias
        self.in_connections = []
        self.lookup_table = {}
        self.is_input = is_input

    def add_connection(self, conn):
        self.in_connections.append(conn)

    # iteratively get value of output neurons
    def get_value(self):
        # squeezes numbers to be sharply between 0 and 1
        def sigmoid5(num):
            return 1/(1+math.e ** (-5 * num))
        SIGMOID = -1000
        valid_connections = list(filter(lambda x: x.expressed, self.in_connections))
        call_stack = []
        value_stack = [0]
        call_stack.append(1) # final sigmoid doesn't have a weight
        call_stack.append(SIGMOID)
        call_stack.append(self.bias)
        count = 0
        # add initial connections to stack from root node
        for conn in valid_connections:
            call_stack.append((conn.in_neuron, conn.weight))
        # run call stack
        while len(call_stack) > 0:
            count += 1
            #print(call_stack)
            #print(value_stack)
            #print("------\n\n\n")
            top = call_stack.pop()
            # at sigmoid, take sigmoid of values and return to stack
            if top == SIGMOID:
                weight = call_stack.pop()
                value = value_stack.pop()
                call_stack.append(weight * sigmoid5(value))
            # at tuple(neuron,weight), break into children
            elif type(top) is tuple:
                weight = top[1]
                neuron = top[0]
                if neuron.is_input:
                    call_stack.append(neuron.value * weight)
                else:
                    valid_connections = list(filter(lambda x: x.expressed, neuron.in_connections))
                    call_stack.append(weight)
                    call_stack.append(SIGMOID)
                    call_stack.append(neuron.bias)
                    value_stack.append(0)
                    for conn in valid_connections:
                        call_stack.append((conn.in_neuron, conn.weight))
            # at number, simply add to value
            else:
                if len(value_stack) == 0:
                    return top
                value = value_stack.pop()
                value_stack.append(top+value)
        return top

    def mutate_bias(self):
        # we don't add biases to input neurons around these parts
        if len(self.in_connections) > 0:
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
