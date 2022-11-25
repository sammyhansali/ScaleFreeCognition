# Run like this: python Experiments/this_file.py
import sys
sys.path.append('.')
from experiment import experiment
from run import run_experiment


# start = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
#         [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
#         [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
#         [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
#         [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
#         [ 0, 1, 0, 0, 0, 0, 0, 1, 0],
#         [ 0, 1, 1, 1, 1, 1, 1, 1, 0],
#         [ 0, 0, 0, 0, 0, 0, 0, 0, 0]]
start = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

ANN_inputs=     [       "molecules", 
                        "local_state",
                        "fitness_score", #each cell should know how far from goal the collective is
                        "pos_x",
                        "pos_y",
                        "direction",
                        "delta",
                        # "finite_reservoir",
                        "bias",
                ]
ANN_inputs.extend(["energy"]*2)
ANN_inputs.extend(["stress"]*2)
ANN_inputs.extend(["state"]*2)
ANN_outputs=    [       "m0_to_send", 
                        "GJ_opening_molecs", 
                        "stress_to_send", 
                        "GJ_opening_stress", 
                        "anxio_to_send", 
                        "apoptosis", 
                        "cell_division",
                        # "reward",
                        # "use_finite_reservoir",
                        "direction",
                ] 
exp = experiment(start, goal, ANN_inputs, ANN_outputs)
exp.nb_gens=int(sys.argv[1])
                        
run_experiment(exp)