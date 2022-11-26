# Run like this: python Experiments/this_file.py
import sys
sys.path.append('.')
from experiment import experiment
from run import run_experiment
#
nb_gens = int(sys.argv[1])
history_length = int(sys.argv[2])
nb_output_molecules = int(sys.argv[3])

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
ANN_inputs.extend(["state"]*history_length)
ANN_inputs.extend(["local_fitness"]*history_length)
ANN_inputs.extend(["global_fitness"]*history_length)


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
for i in range(nb_output_molecules):
        ANN_outputs.append(f"molecule_{i}_to_send")

exp = experiment(start, goal, ANN_inputs, ANN_outputs)
exp.nb_gens = nb_gens
exp.history_length = history_length
exp.nb_output_molecules = nb_output_molecules
                        
run_experiment(exp)