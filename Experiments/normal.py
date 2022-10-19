import sys
sys.path.insert(0, '/home/shansali/Desktop/ScalingCognitionSim')
from experiment import experiment
from run import run_experiment

# French flag
goal = [[ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2],
        [ 1, 1, 1, 3, 3, 3, 2, 2, 2]]

exp = experiment(goal, "french_flag")
exp.nb_gens=1
# exp.nb_inputs=11 # to match the 11 inputs there appears to be
run_experiment(exp)