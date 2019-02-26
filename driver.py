from neat import NEATModel

# train each genome on xor
def test_xor(individual):
    net = individual.neural_net()
    fitness = 4.0
    for x in range(1):
        for y in range(1):
            fitness -= (net.activate(x,y) - x^y) ** 2
    return fitness

model = NEATModel(population_size=5, input_size=2, output_size=1)
model.run(generations=1, fitness_function=test_xor)
