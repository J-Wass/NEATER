from genome import Genome
from gene import NeuronGene, ConnectionGene
import random
import heapq
import configparser

class NEATModel:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.config = config
        self.genomes = []
        self.population_size = int(self.config['Main']['Population Size'])
        self.species = []
        self.generational_talents = int(self.config['Speciation']['Generational Talents'])
        for g in range(self.population_size):
            new_genome = Genome(self.config)
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
    def crossover(self, suitable_genomes):
        while len(self.genomes) < self.population_size - self.generational_talents:
            parent1 = None
            parent2 = None
            # interspecies crossover if this species is too small or by random chance
            if len(suitable_genomes) > 1 and random.uniform(0,1) < 0.01:
                species = random.sample(suitable_genomes,2)
                parent1 = random.choice(species[0])
                parent2 = random.choice(species[1])
            # inner-species crossover
            else:
                chosen_species = random.choice(suitable_genomes)
                if len(chosen_species) > 1:
                    parents = random.sample(chosen_species, 2)
                    parent1 = parents[0]
                    parent2 = parents[1]
                else:
                    continue
            index1 = 0
            index2 = 0
            conn1 = parent1.connection_genes
            conn2 = parent2.connection_genes
            inputs = []
            outputs = []
            connection_genes = []
            neuron_genes = {}
            # adds the connection gene, along with any required neuron genes
            def add_to_genome(gene):
                in_neuron_id = gene.in_neuron.id
                out_neuron_id = gene.out_neuron.id
                in_neuron = None
                out_neuron = None
                # add in_neuron if need be
                if in_neuron_id in neuron_genes:
                    in_neuron = neuron_genes[in_neuron_id]
                else:
                    in_neuron = NeuronGene(id=gene.in_neuron.id, layer=gene.in_neuron.layer, bias=gene.in_neuron.bias, is_input = gene.in_neuron.is_input)
                # add out_neuron if need be
                if out_neuron_id in neuron_genes:
                    out_neuron = neuron_genes[out_neuron_id]
                else:
                    out_neuron = NeuronGene(id=gene.out_neuron.id, layer=gene.out_neuron.layer, bias=gene.out_neuron.bias,  is_output = gene.out_neuron.is_output)
                conn = ConnectionGene(in_neuron, out_neuron, gene.weight, gene.innovation_number)
                out_neuron.add_connection(conn)
                connection_genes.append(conn)
                neuron_genes[in_neuron_id] = in_neuron
                neuron_genes[out_neuron_id] = out_neuron
                if in_neuron.is_input and in_neuron not in inputs:
                    inputs.append(in_neuron)
                if out_neuron.is_output and out_neuron not in outputs:
                    outputs.append(out_neuron)

            while index1 < len(conn1) and index2 < len(conn2):
                gene1 = conn1[index1]
                gene2 = conn2[index2]
                if gene1.innovation_number == gene2.innovation_number:
                    chosen_gene = random.choice([gene1,gene2])
                    add_to_genome(chosen_gene)
                    index1 += 1
                    index2 += 1
                elif gene1.innovation_number < gene2.innovation_number:
                    if parent1.fitness > parent2.fitness:
                        add_to_genome(gene1)
                    index1 += 1
                else:
                    if parent2.fitness > parent1.fitness:
                        add_to_genome(gene2)
                    index2 += 1
            # clean up excess connection genes
            while index1 < len(conn1):
                gene1 = conn1[index1]
                if parent1.fitness > parent2.fitness:
                    add_to_genome(gene1)
                index1 += 1
            while index2 < len(conn2):
                gene2 = conn2[index2]
                if parent2.fitness > parent1.fitness:
                    add_to_genome(gene2)
                index2 += 1

            new_genome = Genome(self.config)
            new_genome.inputs = inputs
            new_genome.outputs = outputs
            new_genome.neuron_genes = neuron_genes
            new_genome.connection_genes = connection_genes
            connections = list(map(lambda x: (x.in_neuron.id, x.out_neuron.id),new_genome.connection_genes))
            self.genomes.append(new_genome)

    def run(self, fitness_function):
        self.fitness_function = fitness_function
        species_threshold = float(self.config['Speciation']['Species Threshold'])
        for gen in range(int(self.config['Main']['Number of Epochs'])):
            print("--Gen {0}--".format(gen))
            self.species.clear()
            # determine fitness (multiprocessing)
            best_genome = None
            for genome in self.genomes:
                # genomes clones from previous epochs already have their fitness
                    genome.fitness =self.fitness_function(genome)
                    if best_genome is None or best_genome.fitness < genome.fitness:
                        best_genome = genome
            print("Best Individual Genome: {0}".format(best_genome))
            if best_genome.fitness > float(self.config['Main']['Target Fitness']):
                print("Solution found, target fitness reached.")
                return best_genome
            for genome in self.genomes:
                found_species = False
                for species in self.species:
                    dist = NEATModel.distance(genome, random.choice(species))
                    if  dist < species_threshold:
                        species.append(genome)
                        found_species = True
                        break
                if not found_species:
                    self.species.append([genome])

            # determine which genomes are suitable for crossover
            suitable_genomes = []
            top_genomes = heapq.nlargest(int(self.generational_talents), self.genomes, key=lambda x: float(x.fitness))
            top_half_genomes = heapq.nlargest(int(len(self.genomes)/2), self.genomes, key=lambda x: float(x.fitness))
            for species in self.species:
                if len(species) > 2:
                    best_of_species = heapq.nlargest(int(len(species)/2), species, key=lambda x: float(x.fitness))
                    suitable_genomes.append(best_of_species)
                if len(species) == 1:
                    endangered_species = species[0]
                    if endangered_species.fitness > top_half_genomes[int(len(self.genomes)/2)-1].fitness:
                        suitable_genomes.append([endangered_species])

            # crossover and mutate
            self.genomes.clear()
            self.crossover(suitable_genomes)
            for genome in self.genomes:
                genome.mutate()

            # add the best genomes unmutated
            for genome in top_genomes:
                self.genomes.append(genome)
