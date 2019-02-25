from gene import Gene
import random
import math

class Genome:
    class NeuralNet:
        def __init__(self, neurons, connections):
            self.neurons = neurons
            self.connections = connections

    def __init__(self, num_inputs, num_output, weight_mutation=0.5, weight_randomize=0.1, neuron_mutation=0.03, connection_mutation=0.05):
        self.weight_mutation = weight_mutation
        self.weight_randomize = weight_randomize
        self.neuron_mutation = neuron_mutation
        self.connection_mutation = connection_mutation
        #TODO: need to keep track of which layer each neuron is in so i can only create forward connections
        self.neuron_genes = list(range(1, num_inputs + num_outputs))
        # initialize connection genes with links from all inputs to all output
        self.connection_genes = []
        for i in range(num_inputs):
            for o in range(num_outputs):
                self.connection_genes.append(Gene(i, o, 1))

    def mutate(self):
        def n_choose_2(n):
            numerator = math.factorial(n)
            denominator = math.factorial(n - 2) * 2
            return numerator/denominator

        # chance of updating weights
        if random.uniform(0,1) < self.weight_mutation_chance:
            for connection_gene in self.connection_genes:
                if random.uniform(0,1) < self.weight_randomize_chance:
                    connection_gene.mutate_weight()
                else
                    connection_gene.randomize_weight()

        # chance of adding a new neuron
        if random.uniform(0,1) < self.neuron_mutation:
            new_neuron_id = len(neuron_genes) + 1
            self.neuron_genes.append(new_neuron_id)
            # choose a random connection to mutate into this new neuron
            mutated_connection = random.choice(self.connection_genes)
            in_neuron = mutated_connection.in_neuron
            out_neuron = mutated_connection.out_neuron
            self.connection_genes.append(Gene(in_neuron,new_neuron_id, 1))
            self.connection_genes.append(Gene(new_neuron_id, out_neuron, mutated_connection.weight))
            mutated_connection.disable()

        # chance of adding a new connection
        if random.uniform(0,1) < self.connection_mutation:
            # make sure we have room for more connections
            maximum_connections = n_choose_2(len(self.neuron_genes) - num_inputs) + num_inputs*(len(self.neuron_genes) - num_inputs)
            if len(self.connection_genes) != maximum_connections:
                # add a random connection between two neurons (not two input neurons tho)
