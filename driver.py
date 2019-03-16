from neat import NEATModel
import math

# solve a simple xor problem
def test_xor(individual):
    fitness = 4.0
    for x in range(2):
        for y in range(2):
            activation = individual.activate([float(x),float(y)])[0]
            fitness -= (activation - float(x^y)) ** 2
    return fitness

# fire a massless/frictionless projectile from a tower with some x and y velocity over a time t
def test_projectile_motion(individual):
    # fitness will be the average distance to the target, 0 is a perfect score
    fitness = 0
    for t in range(1,3): # time
        for x in range(1,3): # x velocity
            for y in range(1,3): # y velocity
                activation = individual.activate([float(t),float(x),float(y)])
                actual_x = x * t
                actual_y = y * t - 9.8 * 0.5 * (t**2)
                fitness -= math.sqrt((actual_x - activation[0])**2 + (actual_y - activation[0])**2)
    fitness /= 27.0
    return fitness

if __name__ == '__main__':
    model = NEATModel(config_file="xor.config")
    best_genome = model.run(fitness_function=test_xor)
