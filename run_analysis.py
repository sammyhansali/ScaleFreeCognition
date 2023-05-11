import pickle
import MultiNEAT as NEAT
import os
import sys
from analysis import sim

# Getting absolute path
# current_dir = sys.argv[1]
current_dir = '/cluster/tufts/levinlab/shansa01/SFC/sfc_results/2023/Apr/27/00:16:39_94'
# current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Loading exp
with open(f"{current_dir}/exp.pickle", "rb") as fp:
    exp = pickle.load(fp)

# Loading Net
net = NEAT.NeuralNetwork()
net.Load(f"{current_dir}/winner_net.txt")
net.Flush()

# Testing stress
exp.preset = [       
    ("preset_wipe_molecule", [1, 1]),
    ("preset_wipe_molecule", [1, 2]),
]

# Simulation
sim(exp, net)