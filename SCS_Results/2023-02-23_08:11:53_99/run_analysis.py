import pickle
import MultiNEAT as NEAT
import os
from analysis import sim

# Getting absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Loading Net
net = NEAT.NeuralNetwork()
net.Load(f"{current_dir}/winner_net.txt")
net.Flush()

# Simulation
sim(net)