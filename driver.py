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
    n = 30
    for i in range (n):
        t = random.uniform(0,4)
        x = 3
        y = 3
        activation = individual.activate([float(t),float(x),float(y)])
        actual_x = x * t
        actual_y = y * t - 9.8 * 0.5 * (t**2)
        fitness -= (math.sqrt((actual_x - activation[0])**2 + (actual_y - activation[1])**2))/n
    return fitness

if __name__ == '__main__':
    model = NEATModel(config_file="projectile.config")
    best_genome = model.run(fitness_function=test_projectile_motion)
    print(best_genome)
    #print("Got {0},{1}, expected {2},{3}".format(ans[0], ans[1], 2.1*3, 2*2.9-9.8*0.5*(3.1**2)))
    #if best_genome is not None:
    #    print("0,0 -> 0 vs {0}".format(best_genome.activate([0,0])[0]))
    #    print("0,1 -> 1 vs {0}".format(best_genome.activate([0,1])[0]))
    #    print("1,0 -> 1 vs {0}".format(best_genome.activate([1,0])[0]))
    #    print("1,1 -> 0 vs {0}".format(best_genome.activate([1,1])[0]))
