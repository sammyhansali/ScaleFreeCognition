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
 
ANN_inputs=     {       
                        "pos_x": 1,
                        "pos_y": 1,
                        "potential": history_length,
                        "cell_type": history_length,
                        "energy": history_length,
                        "molecules": nb_output_molecules*history_length,
                        # "global_fitness": history_length,
                }
nb_ANN_inputs = sum(ANN_inputs.values())
# ANN_inputs.extend(["potential"]*history_length) 
# ANN_inputs.extend(["cell_type"]*history_length)
# ANN_inputs.extend(["energy"]*history_length)
# ANN_inputs.extend(["molecules"]*history_length*nb_output_molecules) 
# ANN_inputs.extend(["global_fitness"]*history_length)

ANN_outputs=    [    
                        # "GJ_opening_molecs",
                ] 
for i in range(nb_output_molecules):
        ANN_outputs.append(f"molecule_{i}_to_send")
        ANN_outputs.append(f"molecule_{i}_GJ")

exp = experiment(start, goal, ANN_inputs, nb_ANN_inputs, ANN_outputs, history_length, nb_gens, e_penalty, nb_output_molecules)

if "potential" in ANN_inputs:
        exp.bioelectric_stimulus = electric
# exp.random_start = True # Turned off for now
new_stim =      [[ -80, -80, -80, -80, -80, -80, -80, -80, -80],
                [ -80, -80, -80, -80, -80, -10, -10, -80, -80],
                [ -80, -50, -80, -80, -80, -80, -10, -80, -80],
                [ -80, -50, -80, -80, -50, -80, -80, -80, -80],
                [ -80, -50, -80, -80, -80, -80, -80, -80, -80],
                [ -80, -50, -80, -80, -50, -80, -80, -80, -80],
                [ -80, -50, -80, -80, -80, -80, -10, -80, -80],
                [ -80, -80, -80, -80, -80, -10, -10, -80, -80],
                [ -80, -80, -80, -80, -80, -80, -80, -80, -80]]
new_goal =     [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [ 1, 1, 1, 1, 1, 3, 3, 1, 1],
                [ 1, 2, 1, 1, 1, 1, 3, 1, 1],
                [ 1, 2, 1, 1, 2, 1, 1, 1, 1],
                [ 1, 2, 1, 1, 1, 1, 1, 1, 1],
                [ 1, 2, 1, 1, 2, 1, 1, 1, 1],
                [ 1, 2, 1, 1, 1, 1, 3, 1, 1],
                [ 1, 1, 1, 1, 1, 3, 3, 1, 1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1]]
new_GJ_0 =      [[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [ 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [ 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
new_stim = new_stim[::-1]
exp.preset =    [       
                        ("preset_bioelectric", [100, new_stim]),
                        ("preset_goal", [100, new_goal]),
                        # ("preset_change_gap_junctions", [10, 0, new_GJ_0]),
                        ("preset_remove_molecule", [100, 0]),
                        ("preset_reset_energy", [250, exp.energy]),
                ]
                        
run_experiment(exp)