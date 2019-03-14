from gene import ConnectionGene, NeuronGene
import random
import math

class Genome:
    def __init__(self, num_inputs, num_outputs, weight_mutation=0.8, neuron_mutation=0.03, connection_mutation=0.05):
        self.fitness = 0
        # chances of the 3 different mutations
        self.weight_mutation = weight_mutation
        self.neuron_mutation = neuron_mutation
        self.connection_mutation = connection_mutation

        # genes that build the genotype
        self.connection_genes = []
        self.neuron_genes = {}
        self.inputs = []
        self.outputs = []
        for i in range(num_inputs):
            neuron = NeuronGene(id=i, layer=0, is_input=True)
            self.neuron_genes[i] = neuron
            self.inputs.append(neuron)
        for o in range(num_outputs):
            neuron = NeuronGene(id=o+num_inputs, layer=1, is_output=True)
            self.neuron_genes[o+num_inputs] = neuron
            self.outputs.append(neuron)
        # initialize connection genes with links from all inputs to all output
        counter = 0
        for i in range(num_inputs):
            for o in range(num_outputs):
                counter += 1
                conn = ConnectionGene(in_neuron=self.neuron_genes[i], out_neuron=self.neuron_genes[o+num_inputs], weight=1, innovation_number=counter)
                self.connection_genes.append(conn)
                conn.out_neuron.add_connection(conn)

    # squeezes numbers to be sharply between 0 and 1
    @staticmethod
    def sigmoid5(num):
        if num >= 0:
            return 1/(1+math.e ** (-5 * num))
        else:
            return (math.e ** (5 * num)) / (1 + math.e ** (5 * num))

    def activate(self, input_list):
        if len(input_list) != len(self.inputs):
            raise Exception('Expected {0} inputs, received {1}.'.format(len(self.inputs), len(input_list)))
        count = 0
        for input in self.inputs:
            #print("{0} inputs".format(id(self)))
            input.value = input_list[count]
            count += 1
        outputs = []
        output_index = 0
        for output in self.outputs:
            SIGMOID = -1000
            valid_connections = list(filter(lambda x: x.expressed, output.in_connections))
            call_stack = []
            value_stack = [0]
            call_stack.append(1) # final sigmoid doesn't have a weight
            call_stack.append(SIGMOID)
            value_stack.append(output.bias)
            count = 0
            # add initial connections to stack from root node
            for conn in valid_connections:
                call_stack.append((conn.in_neuron, conn.weight))
            # run call stack
            while len(call_stack) > 1:
                count += 1
                top = call_stack.pop()
                # at sigmoid, take sigmoid of values and return to stack
                if top == SIGMOID:
                    weight = call_stack.pop()
                    value = value_stack.pop()
                    prev_value = value_stack.pop()
                    value_stack.append(prev_value + weight * Genome.sigmoid5(value))
                # at tuple(neuron,weight), break neuron into children neuron
                else:
                    weight = top[1]
                    neuron = top[0]
                    if neuron.is_input:
                        value = value_stack.pop()
                        value_stack.append(value + neuron.value * weight)
                    else:
                        valid_connections = list(filter(lambda x: x.expressed, neuron.in_connections))
                        call_stack.append(weight)
                        call_stack.append(SIGMOID)
                        value_stack.append(neuron.bias)
                        for conn in valid_connections:
                            call_stack.append((conn.in_neuron, conn.weight))
            outputs.append(value_stack.pop())
        return outputs

    # mutates this genome, either through connection weight or topology
    def mutate(self):
        # chance of updating weights
        if random.uniform(0,1) < self.weight_mutation:
            for connection_gene in self.connection_genes:
                    connection_gene.mutate_weight()

        # chance of adding a new neuron
        if random.uniform(0,1) < self.neuron_mutation:
            # choose a random connection to mutate into this new neuron
            valid_connections = list(filter(lambda x: x.expressed, self.connection_genes))
            if len(valid_connections) > 0:
                mutated_connection = random.choice(valid_connections)
                in_neuron = mutated_connection.in_neuron
                out_neuron = mutated_connection.out_neuron

                # calculate which layer this new neuron is in, create new connections and neuron
                new_neuron_layer = (in_neuron.layer + out_neuron.layer)/2
                new_neuron = NeuronGene(layer=new_neuron_layer)
                new_neuron_id = new_neuron.id

                conn1 = ConnectionGene(in_neuron=in_neuron,out_neuron=new_neuron, weight=1)
                conn2 = ConnectionGene(in_neuron=new_neuron, out_neuron=out_neuron, weight=mutated_connection.weight)
                new_neuron.add_connection(conn1)
                out_neuron.add_connection(conn2)
                self.neuron_genes[new_neuron_id] = new_neuron
                self.connection_genes.append(conn1)
                self.connection_genes.append(conn2)
                mutated_connection.disable()

        # chance of adding a new connection
        if random.uniform(0,1) < self.connection_mutation:
            # choose a random neuron and attempt to connect to some neuron ahead of it
            found_conn = False
            for n in random.sample(list(self.neuron_genes.values()), len(self.neuron_genes.values())):
                possible_targets = list(filter(lambda x: x.layer > n.layer , self.neuron_genes.values()))
                if len(possible_targets) > 0:
                    for t in possible_targets:
                        conns = list(map(lambda x: x.in_neuron, t.in_connections))
                        # make sure this connection doesn't exist
                        if n not in conns:
                            new_connection = ConnectionGene(in_neuron = n, out_neuron = t, weight=1)
                            t.add_connection(new_connection)
                            self.connection_genes.append(new_connection)
                            found_conn = True
                            break
                if found_conn:
                    break

        # chance of mutating a new neuron bias
        if random.uniform(0,1) < self.weight_mutation:
            for neuron in self.neuron_genes.values():
                neuron.mutate_bias()

    def __repr__(self):
        ret_string = '\n-----\nFitness={0}\nNeurons:\n'.format(self.fitness)
        for neuron in self.neuron_genes.values():
            ret_string += "\tid{0}: layer={1} bias={2}\n".format(neuron.id, neuron.layer,neuron.bias)
        ret_string += "\nConnections:\n"
        for connection in self.connection_genes:
            ret_string += "\tInnovation #{0}: id{1}->id{2} weight={3} expressed={4}\n".format(connection.innovation_number, connection.in_neuron.id, connection.out_neuron.id, connection.weight, connection.expressed)
        return ret_string

    def __lt__(self, other):
        return self.fitness < other.fitness
