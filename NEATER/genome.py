from src.gene import ConnectionGene, NeuronGene
import random
import math

class Genome:
    global_id = 0
    @classmethod
    def get_new_global_id(cls):
        """Class method to safely collect global id.

        Can be made thread/process safe in the future if explored.
        Increments cls.global_id for uniqueness.

        Returns
            cls.global_id (int): Unique id for genome

        """
        cls.global_id += 1
        return cls.global_id

    def __init__(self, config):
        """Build genome's initial structure and mutation chances.

        Parameters:
            config (configparser.ConfigParser dictionary): Config file dictionary

        """
        self.fitness = 0
        self.config = config
        mutations = self.config['Mutation']
        self.weight_mutation = float(mutations['Weight Mutation'])
        self.weight_randomization = float(mutations['Weight Randomization'])
        self.neuron_mutation = float(mutations['Neuron Mutation'])
        self.connection_mutation = float(mutations['Connection Mutation'])
        self.aggregation_mutation = float(mutations['Aggregation Mutation'])
        self.aggregation_options = self.config['Activation']['Aggregation Options'].replace(' ','').split(",")
        # genes that build the genotype
        self.connection_genes = []
        self.neuron_genes = {}
        self.inputs = []
        self.outputs = []
        self.id = Genome.get_new_global_id()
        self.history = {}

        num_inputs = int(self.config['Main']['Number of Inputs'])
        num_outputs = int(self.config['Main']['Number of Outputs'])
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

    # activate output neurons
    def activate(self, input_list):
        """Activate genome's phenotype using inputs from input_list.

        Parameters:
            input_list (list of float): Inputs being fed into genome's input neurons

        Returns:
            outputs (list of float): Float list of genome's outputs

        """
        activation_function = self.config['Activation']['Activation Function']
        if len(input_list) != len(self.inputs):
            raise Exception('Expected {0} inputs, received {1}.'.format(len(self.inputs), len(input_list)))
        value_dict = {}
        count = 0
        for input in self.inputs:
            value_dict[input.id] = input_list[count]
            count += 1
        neurons = list(self.neuron_genes.values())
        # order neurons by layer so we can resolve the earlier neurons first
        neurons = sorted(list(filter(lambda x: not x.is_input, neurons)), key=lambda x: x.layer)
        for neuron in neurons:
            activation = neuron.bias
            valid_connections = list(filter(lambda x: x.expressed, neuron.in_connections))
            for conn in valid_connections:
                if neuron.aggregation == 'sum':
                    activation += conn.weight * value_dict[conn.in_neuron.id]
                else:
                    activation *= conn.weight * value_dict[conn.in_neuron.id]
            if activation_function == 'sigmoid':
                value_dict[neuron.id] = Genome.sigmoid(activation, sensitivity=float(self.config['Activation']['Activation Sensitivity']))
            elif activation_function == 'relu':
                value_dict[neuron.id] = Genome.relu(activation)
            else:
                value_dict[neuron.id] = activation
        output_ids = list(map(lambda x: x.id, self.outputs))
        outputs = {k:v for (k,v) in value_dict.items() if k in output_ids}
        return list(outputs.values())

    def mutate(self):
        """Mutuate this genome, changing biases, weights, or topology."""
        # update aggregation option
        if random.uniform(0,1) < self.aggregation_mutation:
            for neuron_gene in self.neuron_genes.values():
                if random.uniform(0,1) < 0.5:
                    neuron_gene.aggregation = random.choice(self.aggregation_options)
        # chance of updating weights
        if random.uniform(0,1) < self.weight_mutation:
            for connection_gene in self.connection_genes:
                    connection_gene.mutate_weight(randomization_chance=self.weight_randomization)

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
            randomized_neurons = random.sample(list(self.neuron_genes.values()), len(self.neuron_genes.values()))
            for n in randomized_neurons:
                # greater layer implies further up the network
                possible_targets = list(filter(lambda x: x.layer > n.layer , self.neuron_genes.values()))
                if len(possible_targets) > 0:
                    for t in possible_targets:
                        conns = list(filter(lambda x: x.expressed, t.in_connections))
                        in_neurons = list(map(lambda x: x.in_neuron, conns))
                        # make sure this connection doesn't exist
                        if n not in in_neurons:
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

    # activation functions
    @staticmethod
    def sigmoid(num, sensitivity):
        """Map values to a stable sigmoid function.

        Parameters:
            sensitivity (float): Activation sensitivity of sigmoid curve

        Returns:
            sigmoid (float): Activated value

        """
        if num >= 0:
            return 1/(1+math.e ** (-1 * sensitivity * num))
        else:
            return (math.e ** (sensitivity * num)) / (1 + math.e ** (sensitivity * num))

    @staticmethod
    def relu(num):
        """Return reLu of a specified input.

        Parameters:
            num (float): reLu input

        Returns:
            output (float): reLu output

        """
        return max(0,num)

    def __repr__(self):
        ret_string = '\n-----\nFitness={0}\nNeurons:\n'.format(self.fitness)
        for neuron in self.neuron_genes.values():
            ret_string += "\tid{0}: layer={1} bias={2} aggregation={3}\n".format(neuron.id, neuron.layer,neuron.bias,neuron.aggregation)
        ret_string += "\nConnections:\n"
        for connection in self.connection_genes:
            ret_string += "\tInnovation #{0}: id{1}->id{2} weight={3} expressed={4}\n".format(connection.innovation_number, connection.in_neuron.id, connection.out_neuron.id, connection.weight, connection.expressed)
        return ret_string

    def __lt__(self, other):
        return self.fitness < other.fitness
