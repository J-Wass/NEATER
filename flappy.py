from NEATER import neat
from ple.games.flappybird import FlappyBird
from ple import PLE
import os
import pickle

flappy_game = None

def test_flappy(individual):
    """Test method for training flappy bird.

    Parameters:
        individual (genome.Genome): Individual being tested

    Returns:
        fitness (float): Float representing this individual's fitness

    """
    global flappy_game
    JUMP = 119 #apparently, action 119 is jump in flappybird
    NOOP = 0
    fitness = 0
    flappy_game.reset_game()
    flappy_game.act(JUMP) #jump to start
    frames = 2500
    for f in range(frames):
        if flappy_game.game_over():
            fitness -= 10
            flappy_game.reset_game()
            flappy_game.act(JUMP)
            continue
        state = flappy_game.getGameState()
        distance = state['next_pipe_dist_to_player']
        next_pipe_height = state['next_next_pipe_top_y']
        pl_height = state['player_y']
        pipe_height = state['next_pipe_top_y']
        jump = individual.activate([distance,next_pipe_height, pl_height,pipe_height])[0]
        if jump < 0.5:
            flappy_game.act(JUMP)
        else:
            flappy_game.act(NOOP)
    return fitness

if __name__ == '__main__':
    # uncomment this block to train headless
    #os.putenv('SDL_VIDEODRIVER', 'fbcon')
    #os.environ["SDL_VIDEODRIVER"] = "dummy"

    # set up flappybird game
    game = FlappyBird()
    # NOTE- if training: set force_fps = true, if testing: set force_fps to false
    flappy_game = PLE(game, fps=30, display_screen=True, force_fps=False)
    flappy_game.init()

    # uncomment this block to train a solution
    #model = neat.NEAT(config_file="flappy.config")
    #best_genome = model.run(fitness_function=test_flappy)
    #pickle.dump( best_genome, open( "flappy", "wb" ) )

    # uncomment this block to test solution
    individual = pickle.load(open("flappy", "rb"))
    fitness = 2000
    flappy_game.reset_game()
    flappy_game.act(119) # apparently action 119 is jump on flappybird
    frames = 3000
    for f in range(frames):
        if flappy_game.game_over():
            fitness -= 30
            flappy_game.reset_game()
            flappy_game.act(119)
            continue
        state = flappy_game.getGameState()
        distance = state['next_pipe_dist_to_player']
        next_pipe_height = state['next_next_pipe_top_y']
        pl_height = state['player_y']
        pipe_height = state['next_pipe_top_y']
        jump = individual.activate([distance,next_pipe_height, pl_height,pipe_height])[0]
        if jump < 0.5:
            flappy_game.act(119)
        else:
            flappy_game.act(0)
