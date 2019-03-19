from neat import NEATModel
import math
import random

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
    n = 1
    for i in range (n):
        t = random.gauss(2,0.33)
        x = random.gauss(3,0.33)
        y = random.gauss(3,0.33)
        g = random.gauss(10,0.33)
        activation = individual.activate([float(t),float(x),float(y),float(g)])
        actual_x = x * t
        actual_y = y * t - g * 0.5 * (t**2)
        fitness -= (math.sqrt((actual_x - activation[0])**2 + (actual_y - activation[0])**2))/n
    return fitness

if __name__ == '__main__':
    model = NEATModel(config_file="xor.config")
    best_genome = model.run(fitness_function=test_xor)
    #print(best_genome)
    #ans = best_genome.activate([2.0,3.0,3.0, 10])
    #print("Got {0},{1}, expected {2},{3}".format(ans[0], ans[1], 2*3, 2*3-10*0.5*(2**2)))
    if best_genome is not None:
        print("0,0 -> 0 vs {0}".format(best_genome.activate([0,0])[0]))
        print("0,1 -> 1 vs {0}".format(best_genome.activate([0,1])[0]))
        print("1,0 -> 1 vs {0}".format(best_genome.activate([1,0])[0]))
        print("1,1 -> 0 vs {0}".format(best_genome.activate([1,1])[0]))
