import sys
sys.path.insert(0, '/home/shansali/Desktop/ScaleFreeCognition')
from experiment import experiment
from run import run_experiment


start = [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]]

goal = [[ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2]]

exp = experiment(start, goal)
exp.nb_gens=1
run_experiment(exp)