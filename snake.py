from src import neat
from ple.games.snake import Snake
from ple import PLE
import os
import pickle

snake_game = None

def test_snake(individual):
    """Test method for training Snake.

    Parameters:
        individual (genome.Genome): Individual being tested

    Returns:
        fitness (float): Float representing this individual's fitness

    """
    global snake_game
    # tbh I don't know if these directions are correct but it doesn't matter
    LEFT = 119
    DOWN = 97
    UP = 100
    RIGHT = 115

    fitness = 0
    snake_game.reset_game()
    frames = 1000
    for f in range(frames):
        if snake_game.game_over():
            snake_game.reset_game()
            continue
        state = snake_game.getGameState()
        sx = state['snake_head_x']
        sy = state['snake_head_y']
        fx = state['food_x']
        fy = state['food_y']
        turns = individual.activate([sx,sy,fx,fy])
        best_turn = max(turns)
        if turns[0] == best_turn:
            fitness += snake_game.act(UP)
        elif turns[1] == best_turn:
            fitness += snake_game.act(LEFT)
        elif turns[2] == best_turn:
            fitness += snake_game.act(RIGHT)
        else:
            fitness += snake_game.act(DOWN)
    return fitness

if __name__ == '__main__':
    # set up flappybird game
    game = Snake(width=400, height=400)
    # NOTE- if training: set force_fps = true, if testing: set force_fps to false
    snake_game = PLE(game, fps=30, display_screen=True, force_fps=False)
    snake_game.init()

    # uncomment this block to train a solution
    #model = neat.NEAT(config_file="snake.config")
    #best_genome = model.run(fitness_function=test_snake)
    #pickle.dump( best_genome, open( "snek", "wb" ) )

    # uncomment this block to test solution
    LEFT = 119
    DOWN = 97
    UP = 100
    RIGHT = 115
    individual = pickle.load(open("snek", "rb"))

    fitness = 0
    snake_game.reset_game()
    frames = 2000
    for f in range(frames):
        if snake_game.game_over():
            snake_game.reset_game()
            continue
        state = snake_game.getGameState()
        sx = state['snake_head_x']
        sy = state['snake_head_y']
        fx = state['food_x']
        fy = state['food_y']
        turns = individual.activate([sx,sy,fx,fy])
        best_turn = max(turns)
        if turns[0] == best_turn:
            snake_game.act(UP)
        elif turns[1] == best_turn:
            snake_game.act(LEFT)
        elif turns[2] == best_turn:
            snake_game.act(RIGHT)
        else:
            snake_game.act(DOWN)
