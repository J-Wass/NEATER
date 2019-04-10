from NEATER import neat
import math
import random

def test_projectile_motion(individual):
    """Test method for training projectile motion.

    Parameters:
        individual (genome.Genome): Individual being tested

    Returns:
        fitness (float): Float representing this individual's fitness

    """
    # fitness will be the average distance to the target, 0 is a perfect score
    fitness = 0
    n = 35
    for i in range (n):
        t = random.uniform(0,4)
        x = random.gauss(2,0.5)
        y = random.gauss(4,0.5)
        activation = individual.activate([float(t),float(x),float(y)])
        actual_x = x * t
        actual_y = y * t - 9.8 * 0.5 * (t**2)
        fitness -= (math.sqrt((actual_x - activation[0])**2 + (actual_y - activation[1])**2))/n
    return fitness

if __name__ == '__main__':
    model = neat.NEAT(config_file="projectile.config")
    best_genome = model.run(fitness_function=test_projectile_motion)
    print(best_genome)
    ans = best_genome.activate([2.1,1.9,4.1])
    print("Got {0},{1}, expected {2},{3}".format(ans[0], ans[1], 1.9*2.1, 4.1*2.1-9.8*0.5*(2.1**2)))
    ans = best_genome.activate([0.2,1.9,3.9])
    print("Got {0},{1}, expected {2},{3}".format(ans[0], ans[1], 1.9*0.2, 3.9*0.2-9.8*0.5*(0.2**2)))
    ans = best_genome.activate([3.6,1.9,4.1])
    print("Got {0},{1}, expected {2},{3}".format(ans[0], ans[1], 1.9*3.6, 4.1*3.6-9.8*0.5*(3.6**2)))
