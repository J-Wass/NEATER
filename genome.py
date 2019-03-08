from gene import ConnectionGene, NeuronGene
import random
import math
import tensorflow as tf

class Genome:
    def __init__(self, num_inputs, num_outputs, weight_mutation=0.8, weight_randomize=0.1, neuron_mutation=0.03, connection_mutation=0.05):
        self.fitness = 0
        # chances of the 4 different mutations
        self.weight_mutation = weight_mutation
        self.weight_randomize = weight_randomize
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
            neuron = NeuronGene(id=o+num_inputs, layer=1)
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
    @staticmethod
    def softmax(inputs):
        total_exponentiated_value = 0
        for i in inputs:
            total_exponentiated_value += math.e ** i
        return list(map(lambda x: (math.e ** x)/total_exponentiated_value ,inputs))
    # squeezes numbers to be sharply between 0 and 1
    @staticmethod
    def sigmoid5(num):
        return 1/(1+math.e ** (-5 * num))

    def activate(self, input_list):
        if len(input_list) != len(self.inputs):
            raise Exception('Expected {0} inputs, received {1}.'.format(len(self.inputs), len(input_list)))
        init_op = self.get_phenotype()
        # start the session
        with tf.Session() as sess:
            sess.run(init_op)
            output = []
            feed = {}
            count = 0
            for tensor in self.input_tensors:
                feed[tensor] = input_list[count]
                count += 1
            for out in self.outputs:
                id = out.id
                out_tensor = self.neuron_tensors[id]
                output.append(sess.run(out_tensor, feed_dict=feed))
        return output

    def get_phenotype(self):
        neuron_tensors = {}
        input_tensors = {}
        for i in self.inputs:
            input_tensors[i.id] = tf.placeholder(tf.float32)
        neurons = [x for x in self.neuron_genes.values() if x not in self.inputs]
        neurons.sort(key=lambda x: x.layer)
        for n in neurons:
            sum_tensors = []
            sum_tensors.append(tf.constant(float(n.bias),dtype=tf.float32))
            for conn in n.in_connections:
                target = conn.in_neuron
                weight = tf.constant(float(conn.weight),dtype=tf.float32)
                if target.id in input_tensors:
                    sum_tensors.append(tf.multiply(weight, input_tensors[target.id]))
                else:
                    sum_tensors.append(tf.multiply(weight, neuron_tensors[target.id]))
            total_tensor = tf.add_n(sum_tensors)
            neuron_tensors[n.id] = tf.sigmoid(total_tensor)
        init_op = tf.global_variables_initializer()
        self.input_tensors = input_tensors.values()
        self.neuron_tensors = neuron_tensors
        return init_op

    # mutates this genome, either through connection weight or topology
    def mutate(self):
        # chance of updating weights
        if random.uniform(0,1) < self.weight_mutation:
            for connection_gene in self.connection_genes:
                if random.uniform(0,1) < self.weight_randomize:
                    connection_gene.mutate_weight()
                else:
                    connection_gene.randomize_weight()

        # chance of adding a new neuron
        if random.uniform(0,1) < self.neuron_mutation:
            # choose a random connection to mutate into this new neuron
            valid_connections = list(filter(lambda x: x.expressed, self.connection_genes))
            if len(valid_connections) > 0:
                mutated_connection = random.choice(valid_connections)
                in_neuron = mutated_connection.in_neuron
                out_neuron = mutated_connection.out_neuron

                # calculate which layer this new neuron is in, create new connections and neuron
                new_neuron_id = len(self.neuron_genes.values())
                new_neuron_layer = (in_neuron.layer + out_neuron.layer)/2
                new_neuron = NeuronGene(new_neuron_id, new_neuron_layer)

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
            # if no neurons are ahead of this neuron, the mutate fails
            in_neuron = random.choice(list(self.neuron_genes.values()))
            possible_targets = list(filter(lambda x: x.layer > in_neuron.layer , self.neuron_genes.values()))
            if len(possible_targets) > 0:
                target = random.choice(possible_targets)
                # if there is already a connection between these two neurons, the mutate fails
                connection_genes = list(filter(lambda x: x.in_neuron == in_neuron and x.out_neuron == target, self.connection_genes))
                if len(connection_genes) == 0:
                    conn = ConnectionGene(in_neuron=in_neuron,out_neuron=target, weight=1)
                    self.connection_genes.append(conn)
                    target.add_connection(conn)

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
