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
# 81-36=45 cells in the goal
goal = [[ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 0, 0, 0]]

#works
# ANN_inputs=     [       "m0_to_send", 
#                         "GJ_opening_molecs", 
#                         "stress_to_send", 
#                         "GJ_opening_stress", 
#                         "anxio_to_send", 
#                         "collective_size", 
#                         # "french_flag",
#                 ] 
ANN_inputs=     [       "m0_to_send", 
                        "GJ_opening_molecs", 
                        "stress_to_send", 
                        "GJ_opening_stress", 
                        "anxio_to_send", 
                        "collective_size", 
                        # "french_flag",
                ] 
ANN_outputs=    [       "m0_to_send", 
                        "GJ_opening_molecs", 
                        "stress_to_send", 
                        "GJ_opening_stress", 
                        "anxio_to_send", 
                        "apoptosis", 
                        "cell_division",
                ] 
exp = experiment(start, goal, ANN_inputs, ANN_outputs)
exp.nb_gens=1
                        
run_experiment(exp)