# 1: Go to this directory -->   /cluster/tufts/levinlab/shansa01/ScaleFreeCognition
# 2: Run like this -->          python Experiments/this_file.py
import sys
sys.path.append('.')
from experiment import experiment
from run import run_experiment

# Command line arguments
nb_gens = int(sys.argv[1])
history_length = int(sys.argv[2])
nb_output_molecules = int(sys.argv[3])

start = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0]]

goal = [[ 0, 0, 0, 3, 3, 3, 0, 0, 0],
        [ 0, 0, 0, 3, 3, 3, 0, 0, 0],
        [ 0, 0, 0, 3, 3, 3, 0, 0, 0],
        [ 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [ 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [ 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [ 0, 0, 0, 3, 3, 3, 0, 0, 0],
        [ 0, 0, 0, 3, 3, 3, 0, 0, 0],
        [ 0, 0, 0, 3, 3, 3, 0, 0, 0]]
 
ANN_inputs=     [       
                        "pos_x",
                        "pos_y",
                        "goal",
                        # "finite_reservoir",
                        "bias",
                ]
ANN_inputs.extend(["molecules"]*history_length*nb_output_molecules) 
ANN_inputs.extend(["energy"]*history_length)
ANN_inputs.extend(["stress"]*history_length)
ANN_inputs.extend(["state"]*history_length)
ANN_inputs.extend(["global_fitness"]*history_length)
ANN_inputs.extend(["direction"]*history_length)


ANN_outputs=    [    
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
for i in range(nb_output_molecules):
        ANN_outputs.append(f"molecule_{i}_to_send")

exp = experiment(start, goal, ANN_inputs, ANN_outputs)
exp.nb_gens = nb_gens
exp.history_length = history_length
                        
run_experiment(exp)