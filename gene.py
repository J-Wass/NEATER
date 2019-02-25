import random

# connection gene that describes the weight between two neurons
class Gene:
    innovation_number = 1
    def __init__(self, in_neuron, out_neuron, weight):
        self.in_neuron = in_neuron
        self.out_neuron = out_neuron
        self.weight = weight
        self.expressed = True
        self.innovation_number = innovation_number
        innovation_number += 1

    def mutate_weight(self):
        self.weight *= random.uniform(0.5, 1.5)

    def randomize_weight(self):
        self.weight = random.uniform(-2, 2)

    def disable(self):
        self.expressed = False
