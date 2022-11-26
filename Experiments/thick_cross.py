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
 
ANN_inputs=     [       #"molecules", 
                        # "local_fitness",
                        # "global_fitness", # aka fitness score
                        "pos_x",
                        "pos_y",
                        "direction",
                        "delta",
                        # "finite_reservoir",
                        "bias",
                ]
# ANN_inputs.extend(["energy"]*2)
# ANN_inputs.extend(["stress"]*2)
ANN_inputs.extend(["state"]*5)
ANN_inputs.extend(["local_fitness"]*5)
ANN_inputs.extend(["global_fitness"]*5)
ANN_outputs=    [       #"m0_to_send", 
                        # "GJ_opening_molecs", 
                        # "stress_to_send", 
                        # "GJ_opening_stress", 
                        # "anxio_to_send", 
                        "apoptosis", 
                        "cell_division",
                        # "reward",
                        # "use_finite_reservoir",
                        "direction",
                ] 
# Maybe make a molecules dictionary?
# Keys: 1,2,3, etc
# Values: timeseries lists of histories of molecules
# Get states by summing 0th index of each value[]

exp = experiment(start, goal, ANN_inputs, ANN_outputs)
exp.nb_gens=int(sys.argv[1])
                        
run_experiment(exp)