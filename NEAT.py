from genome import Genome
import random

class NEATModel:
    def __init__(self, population_size, input_size, output_size):
        self.genomes = []
        self.population_size = population_size
        self.species = []
        for g in range(self.population_size):
            new_genome = Genome(num_inputs=input_size, num_outputs=output_size)
            new_genome.mutate()
            new_genome.ave_weight = 0
            for conn in new_genome.connection_genes:
                new_genome.ave_weight += conn.weight / len(new_genome.connection_genes)
            self.genomes.append(new_genome)

    def run(self, generations, fitness_function, species_threshold = 3.0
        # function to determine the "distance" between two genomes
        def distance(genome1, genome2):
            disjoint_genes = 0
            similar_genes = 0
            similar_weight = 0
            innovation_lookup = {}
            for connection in list(set(genome1.connection_genes) | set(genome2.connection_genes)):
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
            return disjoint_genes / N + similar_weight / similar_genes * 0.5

        self.fitness_function = fitness_function
        for gen in range(generations):
            for genome in self.genomes:
                # determine fitness of each genome
                genome.fitness = fitness_function(genome)
                # speciate each genome
                if len(self.species) == 0:
                    self.species.append([genome])
                else:
                    found_species = False
                    for species in self.species:
                        if distance(genome, random.choice(species)) < species_threshold:
                            species.append(genome)
                            found_species = True
                            break
                    if not found_species:
                        self.species.append([genome])
            # determine which genomes are suitable for crossover
            suitable_genomes = []
            for species in self.species:
                # fitness tournament to find suitable genomes
                if len(species) > 2:
                    for x in range(len(species)/2):
                        genomes = random.sample(species, 2)
                        best_genome = max(genomes, key=lambda x: x.fitness)
                        suitable_genomes.add(best_genome)
        #crossover until len(genomes) = population_size
        #mutate each genome
