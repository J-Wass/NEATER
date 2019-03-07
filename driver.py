from neat import NEATModel

# individual is an instance of genome.Genome
def test_xor(individual):
    #print(individual)
    fitness = 4.0
    for x in range(2):
        for y in range(2):
            act = individual.activate([x,y])[0]
            try:
                fitness -= (act - float(x^y)) ** 2
            except TypeError:
                print(individual)
                for gene in individual.neuron_genes.values():
                    print(gene.value)
            #print("{0} vs {1}".format(act, float(x^y)))
    #print(fitness)
    return fitness

model = NEATModel(population_size=300, input_size=2, output_size=1)
model.run(generations=50, fitness_function=test_xor)
