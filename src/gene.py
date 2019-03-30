import random

class NeuronGene:
    global_id = 1
    @classmethod
    def get_global_id(cls):
        """Class method, get global id for neuron genes.

        Made to be thread/process safe in the future.

        Returns:
            cls.global_id (int): Unique id for a neuron gene.

        """
        return cls.global_id
    @classmethod
    def increment_global_id(cls):
        """Increment the global id for all neuron genes."""
        cls.global_id += 1

    def __init__(self, id=None, layer=0, bias = 0, is_input = False, is_output = False, aggregation = 'sum'):
        """Initialize a neuron gene.

        Parameters:
            id (int optional): Specific id of neuron gene. Automatically assigned if None
            layer (float optional): Location of neuron gene in phenotype
            bias (float optional): Bias of neuron gene in phenotype
            is_input (bool optional): True if neuron is an input_neuron
            is_output (bool optional): True if neuron is an output_neuron
            aggregation (string optional): Type of aggregation neuron uses during activation

        """
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
        """Add connection gene to self.in_connections.

        Parameters:
            conn (gene.ConnectionGene): ConnectionGene being added to neuron

        """
        self.in_connections.append(conn)

    def mutate_bias(self):
        """Mutate self.bias."""
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
        """Get current innovation number for connections.

        Returns:
            cls.innovation_number (int): Current innovation number for all connections.

        """
        return cls.innovation_number
    @classmethod
    def increment_innovation_number(cls):
        """Increment the current innovation number."""
        cls.innovation_number += 1

    def __init__(self, in_neuron, out_neuron, weight, innovation_number = None):
        """Initialize a connection gene.

        Parameters:
            in_neuron (gene.NeuronGene): In coming neuron gene for this connection
            out_neuron (gene.NeuronGene): Out going neuron gene for this connection
            weight (float): Weight used in phenotype
            innovation_number (int): Unique identifier for this connection gene

        """
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
        """Mutate self.weight.

        Parameters:
            randomization_chance (float): Chance self.weight will be randomized

        """
        self.weight *= random.uniform(0.5, 1.5)
        # on rare occasion, flip sign of weight
        if random.uniform(0,1) < randomization_chance:
            self.weight = random.uniform(-2,2)

    def disable(self):
        """Disable this connection gene."""
        self.expressed = False

    def __repr__(self):
        ret_string = "Innovation{0}: \n\tin_neuron={1} \n\tout_neuron={1} \n\tweight={2} \n\texpressed={3} \n\n".format(self.innovation_number, self.in_neuron.id,self.out_neuron.id, self.weight, self.expressed)
        return ret_string
