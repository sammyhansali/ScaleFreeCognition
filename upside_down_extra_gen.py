import pickle
import MultiNEAT as NEAT
import os
import sys
from analysis import sim


# Getting absolute path
file_name = sys.argv[1]
print(file_name)
current_dir = f"/cluster/tufts/levinlab/shansa01/SFC/sfc_results/{file_name}"
# os.chdir(current_dir)

# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)

# Loading exp
with open(f"{current_dir}/exp.pickle", "rb") as fp:
    exp = pickle.load(fp)

# Loading Net
net = NEAT.NeuralNetwork()
net.Load(f"{current_dir}/winner_net.txt")
net.Flush()

### Upside down face
new_stim = exp.bioelectric_stimulus[::-1]
new_goal = exp.goal[::-1]

# ### Clockwise turn
# new_stim=       [
#     [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
#     [ -90, -90, -90, -90, -90, -40, -40, -90, -90],
#     [ -90, -65, -90, -90, -90, -90, -40, -90, -90],
#     [ -90, -65, -90, -90, -15, -90, -90, -90, -90],
#     [ -90, -65, -90, -90, -90, -90, -90, -90, -90],
#     [ -90, -65, -90, -90, -15, -90, -90, -90, -90],
#     [ -90, -65, -90, -90, -90, -90, -40, -90, -90],
#     [ -90, -90, -90, -90, -90, -40, -40, -90, -90],
#     [ -90, -90, -90, -90, -90, -90, -90, -90, -90]
# ][::-1]
# new_goal =  [
#     [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [ 1, 1, 1, 1, 1, 3, 3, 1, 1],
#     [ 1, 2, 1, 1, 1, 1, 3, 1, 1],
#     [ 1, 2, 1, 1, 4, 1, 1, 1, 1],
#     [ 1, 2, 1, 1, 1, 1, 1, 1, 1],
#     [ 1, 2, 1, 1, 4, 1, 1, 1, 1],
#     [ 1, 2, 1, 1, 1, 1, 3, 1, 1],
#     [ 1, 1, 1, 1, 1, 3, 3, 1, 1],
#     [ 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ][::-1]

# ### New looking face - WORKS
# new_stim = [    
#     [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
#     [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
#     [ -90, -40, -40, -90, -90, -90, -40, -40, -90],
#     [ -90, -40, -90, -90, -90, -90, -90, -40, -90],
#     [ -90, -90, -90, -90, -15, -90, -90, -90, -90],
#     [ -90, -90, -90, -90, -15, -90, -90, -90, -90],
#     [ -90, -90, -65, -90, -90, -90, -65, -90, -90],
#     [ -90, -90, -90, -65, -65, -65, -90, -90, -90],
#     [ -90, -90, -90, -90, -90, -90, -90, -90, -90]
# ][::-1]
# new_goal = [        
#     [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [ 1, 3, 3, 1, 1, 1, 3, 3, 1],
#     [ 1, 3, 1, 1, 1, 1, 1, 3, 1],
#     [ 1, 1, 1, 1, 4, 1, 1, 1, 1],
#     [ 1, 1, 1, 1, 4, 1, 1, 1, 1],
#     [ 1, 1, 2, 1, 1, 1, 2, 1, 1],
#     [ 1, 1, 1, 2, 2, 2, 1, 1, 1],
#     [ 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ][::-1]

exp.preset =    [       
    ("preset_bioelectric", [100, new_stim]),
    ("preset_goal", [100, new_goal]),
    # ("preset_change_gap_junctions", [10, 0, new_GJ_0]),
    # ("preset_remove_molecule", [100, 0]),
    ("preset_reset_energy", [100, exp.energy]),
]

# Simulation
sim(exp, net)