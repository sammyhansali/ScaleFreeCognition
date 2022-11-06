# Run like this: python Experiments/this_file.py
import sys
sys.path.append('.')
from experiment import experiment
from run import run_experiment
#

start = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0]]

goal = [[ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0]]

exp = experiment(start, goal)
exp.nb_gens=250
run_experiment(exp)