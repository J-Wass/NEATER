from genome import Genome
from gene import NeuronGene, ConnectionGene
import random
import heapq

class NEATModel:
    def __init__(self, population_size, input_size, output_size):
        self.genomes = []
        self.population_size = population_size
        self.species = []
        for g in range(self.population_size):
            new_genome = Genome(num_inputs=input_size, num_outputs=output_size)
            new_genome.mutate()
            self.genomes.append(new_genome)

    # helper function to determine the "distance" between two genomes
    @staticmethod
    def distance(genome1, genome2):
        disjoint_genes = 0
        similar_genes = 0
        similar_weight = 0
        innovation_lookup = {}
        genome1_conn = list(filter(lambda x: x.expressed, genome1.connection_genes))
        genome2_conn = list(filter(lambda x: x.expressed, genome2.connection_genes))
        for connection in list(set(genome1_conn) | set(genome2_conn)):
            if connection.innovation_number in innovation_lookup:
                disjoint_genes -= 1
                similar_genes += 1
                weight_difference = abs(innovation_lookup[connection.innovation_number] - connection.weight)
                similar_weight += weight_difference
            else:
                innovation_lookup[connection.innovation_number] = connection.weight
                disjoint_genes += 1
        N = max(len(genome1.connection_genes), len(genome2.connection_genes))
        # for small genomes, just use N = 1
        if N < 20:
            N = 1
        dist = disjoint_genes / N
        if similar_genes > 0:
            dist += similar_weight / similar_genes * 0.5
        return dist

    # helper function to perform genetic cross over
    # this is the worst function I have ever written
    def crossover(self, suitable_genomes):
        population = 1
        while population <= self.population_size:
            chosen_species = random.choice(suitable_genomes)
            parents = [None,None]
            # if species has 1 population or by chance, perform interspecies crossover
            if random.uniform(0,1) < 0.01 or len(chosen_species) == 1:
                # if another species is available, perform interspecies crossover
                if len(suitable_genomes) > 1:
                    parents[0] = random.choice(chosen_species)
                    suitable_genomes.remove(chosen_species)
                    new_species = random.choice(suitable_genomes)
                    parents[1] = random.choice(new_species)
                else:
                    parents = random.sample(chosen_species,2)
            # usually just perform crossover from within your species
            else:
                parents = random.sample(chosen_species,2)

            population += 1
            parent1 = parents[0]
            parent2 = parents[1]
            p1_gene_index = 0
            p2_gene_index = 0
            connection_genes = []
            neuron_genes = {}
            inputs = []
            outputs = []
            neuron_counter = 0

            # add initial neuron genes
            for i in parent1.inputs:
                neuron = NeuronGene(id=neuron_counter,layer = 0, is_input=True)
                neuron_genes[neuron_counter] = neuron
                inputs.append(neuron)
                neuron_counter += 1
            for o in parent1.outputs:
                neuron = NeuronGene(id=neuron_counter,layer = 1)
                neuron_genes[neuron_counter] = neuron
                outputs.append(neuron)
                neuron_counter += 1

            # add all connection genes and added neurons
            while p1_gene_index < len(parent1.connection_genes) and p2_gene_index < len(parent2.connection_genes):
                p1 = parent1.connection_genes[p1_gene_index]
                p2 = parent2.connection_genes[p2_gene_index]
                # randomly choose one of the alleles if they are the same gene
                if p1.innovation_number == p2.innovation_number:
                    new_gene_expressed = True
                    if not p1.expressed or not p2.expressed:
                        new_gene_expressed = random.uniform(0,1) < 0.25
                    chosen_connection = random.choice([p1, p2])
                    in_neuron = None
                    out_neuron = None
                    if chosen_connection.in_neuron.id in neuron_genes:
                        in_neuron = neuron_genes[chosen_connection.in_neuron.id]
                    else:
                        in_neuron = NeuronGene(id=neuron_counter, layer=chosen_connection.in_neuron.layer, bias=chosen_connection.in_neuron.bias)
                        neuron_genes[neuron_counter] = in_neuron
                        neuron_counter += 1
                    if chosen_connection.out_neuron.id in neuron_genes:
                        out_neuron = neuron_genes[chosen_connection.out_neuron.id]
                    else:
                        out_neuron = NeuronGene(id=neuron_counter, layer=chosen_connection.out_neuron.layer, bias=chosen_connection.out_neuron.bias)
                        neuron_genes[neuron_counter] = out_neuron
                        neuron_counter += 1
                    new_connection = ConnectionGene(in_neuron, out_neuron, chosen_connection.weight, chosen_connection.innovation_number)
                    new_connection.expressed = new_gene_expressed
                    out_neuron.add_connection(new_connection)
                    connection_genes.append(new_connection)
                    p2_gene_index += 1
                    p1_gene_index += 1

                # take a disjoint gene if parent is fitter
                elif p1.innovation_number > p2.innovation_number and parent2.fitness >= parent1.fitness:
                    in_neuron = None
                    out_neuron = None
                    if p2.in_neuron.id in neuron_genes:
                        in_neuron = neuron_genes[p2.in_neuron.id]
                    else:
                        in_neuron = NeuronGene(id=neuron_counter, layer=p2.in_neuron.layer, bias=p2.in_neuron.bias)
                        neuron_genes[neuron_counter] = in_neuron
                        neuron_counter += 1
                    if p2.out_neuron.id in neuron_genes:
                        out_neuron = neuron_genes[p2.out_neuron.id]
                    else:
                        out_neuron = NeuronGene(id=neuron_counter, layer=p2.out_neuron.layer, bias=p2.out_neuron.bias)
                        neuron_genes[neuron_counter] = out_neuron
                        neuron_counter += 1
                    new_connection = ConnectionGene(in_neuron, out_neuron, p2.weight, p2.innovation_number)
                    out_neuron.add_connection(new_connection)
                    connection_genes.append(new_connection)
                    p2_gene_index += 1
                elif parent1.fitness > parent2.fitness:
                    in_neuron = None
                    out_neuron = None
                    if p1.in_neuron.id in neuron_genes:
                        in_neuron = neuron_genes[p1.in_neuron.id]
                    else:
                        in_neuron = NeuronGene(id=neuron_counter,layer=p1.in_neuron.layer, bias=p1.in_neuron.bias)
                        neuron_genes[neuron_counter] = in_neuron
                        neuron_counter += 1
                    if p1.out_neuron.id in neuron_genes:
                        out_neuron = neuron_genes[p1.out_neuron.id]
                    else:
                        out_neuron = NeuronGene(id=neuron_counter, layer=p1.out_neuron.layer, bias=p1.out_neuron.bias)
                        neuron_genes[neuron_counter] = out_neuron
                        neuron_counter += 1
                    new_connection = ConnectionGene(in_neuron, out_neuron, p1.weight, p1.innovation_number)
                    out_neuron.add_connection(new_connection)
                    connection_genes.append(new_connection)
                    p1_gene_index += 1
                else:
                    # gene is disjoint but not from fitter parent, just continue
                    if p1.innovation_number > p2.innovation_number:
                        p2_gene_index += 1
                    else:
                        p1_gene_index += 1
            # take excess genes from more fit parent
            while p1_gene_index < len(parent1.connection_genes) and parent1.fitness >= parent2.fitness:
                connection = parent1.connection_genes[p1_gene_index]
                in_neuron = None
                out_neuron = None
                if connection.in_neuron.id in neuron_genes:
                    in_neuron = neuron_genes[connection.in_neuron.id]
                else:
                    in_neuron = NeuronGene(id=neuron_counter, layer=connection.in_neuron.layer, bias=connection.in_neuron.bias)
                    neuron_genes[neuron_counter] = in_neuron
                    neuron_counter += 1
                if connection.out_neuron.id in neuron_genes:
                    out_neuron = neuron_genes[connection.out_neuron.id]
                else:
                    out_neuron = NeuronGene(id=neuron_counter, layer=connection.out_neuron.layer, bias=connection.out_neuron.bias)
                    neuron_genes[neuron_counter] = out_neuron
                    neuron_counter += 1
                new_connection = ConnectionGene(in_neuron, out_neuron, connection.weight, connection.innovation_number)
                out_neuron.add_connection(new_connection)
                connection_genes.append(new_connection)
                p1_gene_index += 1

            while p2_gene_index < len(parent2.connection_genes) and parent2.fitness >= parent1.fitness:
                connection = parent2.connection_genes[p2_gene_index]
                in_neuron = None
                out_neuron = None
                if connection.in_neuron.id in neuron_genes:
                    in_neuron = neuron_genes[connection.in_neuron.id]
                else:
                    in_neuron = NeuronGene(id=neuron_counter, layer=connection.in_neuron.layer, bias=connection.in_neuron.layer)
                    neuron_genes[neuron_counter] = in_neuron
                    neuron_counter += 1
                if connection.out_neuron.id in neuron_genes:
                    out_neuron = neuron_genes[connection.out_neuron.id]
                else:
                    out_neuron = NeuronGene(id=neuron_counter, layer=connection.out_neuron.layer, bias=connection.out_neuron.bias)
                    neuron_genes[neuron_counter] = out_neuron
                    neuron_counter += 1
                new_connection = ConnectionGene(in_neuron, out_neuron, connection.weight, connection.innovation_number)
                out_neuron.add_connection(new_connection)
                connection_genes.append(new_connection)
                p2_gene_index += 1
            g = suitable_genomes[0][0]
            new_genome = Genome(len(inputs), len(outputs), g.weight_mutation, g.weight_randomize, g.neuron_mutation, g.connection_mutation)
            new_genome.inputs = inputs
            new_genome.outputs = outputs
            new_genome.neuron_genes = neuron_genes
            new_genome.connection_genes = connection_genes
            self.genomes.append(new_genome)

    def run(self, generations, fitness_function, species_threshold = 3.0):
        self.fitness_function = fitness_function
        top = None
        for gen in range(generations):
            self.species = []
            for genome in self.genomes:
                # determine fitness of each genome, then speciate each genome
                genome.fitness = self.fitness_function(genome)
                if top is None or genome.fitness > top.fitness:
                    top = genome
                if len(self.species) == 0:
                    self.species.append([genome])
                else:
                    found_species = False
                    for species in self.species:
                        dist = NEATModel.distance(genome, random.choice(species))
                        if  dist < species_threshold:
                            species.append(genome)
                            found_species = True
                            break
                    if not found_species:
                        self.species.append([genome])
            print(top.fitness)
            # determine which genomes are suitable for crossover
            self.genome = []
            suitable_genomes = []
            for species in self.species:
                # use a heap to get the top individuals in each species
                if len(species) == 1:
                    best_one = heapq.nlargest(1, species, key=lambda x: float(x.fitness))
                    self.genomes.append(best_one[0])
                    suitable_genomes.append(best_one)
                else:
                    best_of_species = heapq.nlargest(int(len(species)/2), species, key=lambda x: float(x.fitness))
                    suitable_genomes.append(best_of_species)
                    self.genomes.append(best_of_species[0])

            if len(suitable_genomes) == 0:
                raise Exception("Population is too speciated, either lower mutation chances or increase speciation distances")
            # crossover and mutate
            self.crossover(suitable_genomes)
            for genome in self.genomes:
                genome.mutate()
