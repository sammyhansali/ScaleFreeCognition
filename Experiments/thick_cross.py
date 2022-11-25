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
 
ANN_inputs=     [       "molecules", 
                        # "energy", 
                        # "energyt1", 
                        # "stress", 
                        # "stresst1", 
                        # "state_goal"
                        # "state",
                        # "statet1",
                        # "local_geometrical_frustration",
                        "local_state",
                        # "collective_size", 
                        # "french_flag",
                        "pos_x",
                        "pos_y",
                        "finite_reservoir",
                        "fitness_score", #each cell should know how far from goal they are.
                        "bias",
                ]
ANN_inputs.extend(["energy"]*10)
ANN_inputs.extend(["stress"]*10)
ANN_inputs.extend(["state"]*10)
ANN_outputs=    [       "m0_to_send", 
                        "GJ_opening_molecs", 
                        "stress_to_send", 
                        "GJ_opening_stress", 
                        "anxio_to_send", 
                        "apoptosis", # prob doesn't need this function, since will die from energy loss if it is in the wrong spot...
                        "cell_division",
                        "reward",
                        "use_finite_reservoir",
                ] 
exp = experiment(start, goal, ANN_inputs, ANN_outputs)
exp.nb_gens=int(sys.argv[1])
                        
run_experiment(exp)