from genome import Genome

class NEAT:
    def __init__(self, population_size, input_size, output_size):
        self.genomes = []
        for g in range(population_size):
            new_genome = Genome(input_size, output_size)
            new_genome.mutate()
            self.genomes.append(new_genome)


    def run(self, generations, fitness_function):
        self.fitness_function = fitness_function
        #for x in range(generations)
        #get fitness of each genome
        #speciate each genome
        #kill bottom 50% of each genome
        #crossover until len(genomes) = population_size
        #mutate each genome
