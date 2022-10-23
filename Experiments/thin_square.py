import sys
sys.path.insert(0, '/home/shansali/Desktop/ScalingCognitionSim')
from experiment import experiment
from run import run_experiment


start = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0]]

goal = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0]]

exp = experiment(start, goal)
exp.nb_gens=1
run_experiment(exp)