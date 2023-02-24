import pickle
import MultiNEAT as NEAT
import os
from analysis import sim

# Getting absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Loading exp
with open(f"{current_dir}/exp.pickle", "rb") as fp:
    exp = pickle.load(fp)

# Loading Net
net = NEAT.NeuralNetwork()
net.Load(f"{current_dir}/winner_net.txt")
net.Flush()

# Simulation
sim(exp, net)