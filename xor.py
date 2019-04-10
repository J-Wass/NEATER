from NEATER import neat

def test_xor(individual):
    """Test method for training bitwise xor.

    Parameters:
        individual (genome.Genome): Individual being tested

    Returns:
        fitness (float): Float representing this individual's fitness

    """
    fitness = 4.0
    for x in range(2):
        for y in range(2):
            activation = individual.activate([float(x),float(y)])[0]
            fitness -= (activation - float(x^y)) ** 2
    return fitness

if __name__ == '__main__':
    model = neat.NEAT(config_file="xor.config")
    best_genome = model.run(fitness_function=test_xor)
    print(best_genome)
    print("0,0 -> 0 vs {0}".format(best_genome.activate([0,0])[0]))
    print("0,1 -> 1 vs {0}".format(best_genome.activate([0,1])[0]))
    print("1,0 -> 1 vs {0}".format(best_genome.activate([1,0])[0]))
    print("1,1 -> 0 vs {0}".format(best_genome.activate([1,1])[0]))
