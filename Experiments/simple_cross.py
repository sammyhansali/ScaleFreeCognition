import sys
sys.path.insert(0, '/home/shansali/Desktop/ScalingCognitionSim')
from experiment import experiment
from run import run_experiment

# Simple Cross
goal = [[ 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0]]

exp = experiment(goal, "french_flag")
run_experiment(exp)