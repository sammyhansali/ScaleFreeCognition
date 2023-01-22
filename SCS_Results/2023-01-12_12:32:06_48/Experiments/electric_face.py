# 1: Go to this directory -->   /cluster/tufts/levinlab/shansa01/ScaleFreeCognition
# 2: Run like this -->          python Experiments/this_file.py
import sys
sys.path.append('.')
from experiment import experiment
from run import run_experiment

# Command line arguments
nb_gens = int(sys.argv[1])              # 250
history_length = int(sys.argv[2])       # 2
nb_output_molecules = int(sys.argv[3])  # 3
e_penalty = float(sys.argv[4])          # 0.95

## Deranged Face
start =    [[ 3, 3, 1, 1, 1, 1, 1, 1, 1],
            [ 3, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 2, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 2, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 2, 2, 2, 2, 2, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 3, 3],
            [ 1, 1, 1, 1, 1, 1, 1, 3, 1]]
start = start[::-1]     # Needed for visualization since grid placement at 0,0 is bottom left instead of top left
## Target Face
goal =     [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 3, 3, 1, 1, 1, 3, 3, 1],
            [ 1, 3, 1, 1, 1, 1, 1, 3, 1],
            [ 1, 1, 1, 2, 1, 2, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 2, 2, 2, 2, 2, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1]]
goal = goal[::-1]     # Needed for visualization since grid placement at 0,0 is bottom left instead of top left

# 3 = Retinal Cell
# 2 = Epithelial (for nose and mouth)
# 1 = Skin

electric = [[ -80, -80, -80, -80, -80, -80, -80, -80, -80],
            [ -80, -80, -80, -80, -80, -80, -80, -80, -80],
            [ -80, -10, -10, -80, -80, -80, -10, -10, -80],
            [ -80, -10, -80, -80, -80, -80, -80, -10, -80],
            [ -80, -80, -80, -50, -80, -50, -80, -80, -80],
            [ -80, -80, -80, -80, -80, -80, -80, -80, -80],
            [ -80, -80, -80, -80, -80, -80, -80, -80, -80],
            [ -80, -80, -50, -50, -50, -50, -50, -80, -80],
            [ -80, -80, -80, -80, -80, -80, -80, -80, -80]]
electric = electric[::-1]
# -40 or higher = Hyperpolarized        (Retina)
# -50 = Neutral                         (Epithelial)
# -60 or less = Depolarized             (Skin)
 
ANN_inputs=     [       
                        "bias",
                        "pos_x",
                        "pos_y",
                ]
ANN_inputs.extend(["potential"]*history_length) 
ANN_inputs.extend(["cell_type"]*history_length)
ANN_inputs.extend(["energy"]*history_length)
ANN_inputs.extend(["molecules"]*history_length*nb_output_molecules) 
ANN_inputs.extend(["global_fitness"]*history_length)

ANN_outputs=    [    
                        # "GJ_opening_molecs",
                ] 
for i in range(nb_output_molecules):
        ANN_outputs.append(f"molecule_{i}_to_send")
        ANN_outputs.append(f"molecule_{i}_GJ")

exp = experiment(start, goal, ANN_inputs, ANN_outputs)
exp.nb_gens = nb_gens
exp.history_length = history_length
exp.e_penalty = e_penalty
if "potential" in ANN_inputs:
        exp.bioelectric_stimulus = electric
# exp.random_start = True # Turned off for now
exp.preset = "preset1"
                        
run_experiment(exp)