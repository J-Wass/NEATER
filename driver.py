from neat import NEATModel

# train each genome on xor
def test_xor(individual):
    fitness = 4.0
    for x in range(2):
        for y in range(2):
            act = individual.activate([x,y])[0]
            fitness -= (act - float(x^y)) ** 2
            #print("{0} vs {1}".format(act, float(x^y)))
    return fitness

model = NEATModel(population_size=300, input_size=2, output_size=1)
model.run(generations=300, fitness_function=test_xor)
