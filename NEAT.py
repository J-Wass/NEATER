from genome import Genome

class NEATModel:
    def __init__(self, population_size, input_size, output_size):
        self.genomes = []
        for g in range(population_size):
            new_genome = Genome(num_inputs=input_size, num_output=output_size)
            new_genome.mutate()
            self.genomes.append(new_genome)
            print(new_genome)

    def run(self, generations, fitness_function):
        self.fitness_function = fitness_function
        for x in range(generations):
            for genome in self.genomes:
                genome.fitness = fitness_function(genome)
                print(genome.fitness)
        #get fitness of each genome
        #speciate each genome
        #kill bottom 50% of each genome
        #crossover until len(genomes) = population_size
        #mutate each genome
