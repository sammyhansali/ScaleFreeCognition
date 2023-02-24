import pickle
import MultiNEAT as NEAT
from analysis import sim

with open("exp.pickle", "rb") as fp:
    exp = pickle.load(fp)
best_genome = NEAT.Genome("best_genome.txt")

sim(exp, best_genome)