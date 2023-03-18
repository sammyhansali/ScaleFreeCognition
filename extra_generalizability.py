import pickle
import MultiNEAT as NEAT
import os
from analysis import sim
from multicellularity.model_for_analysis import Multicellularity_model
from random_faces import RandomFaces
import sys


### JUST HAVE TO BE ABLE TO EXTRA GENERALIZE ONCE - SINCE THERE IS TOO MUCH ERROR/VARIANCE

file_names='''
2023/Mar/13/22:25:14_96
'''.split()


def getOptions():
    new_stims = [
        # Clockwise turn
        [[ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -90, -90, -90, -40, -40, -90, -90],
        [ -90, -65, -90, -90, -90, -90, -40, -90, -90],
        [ -90, -65, -90, -90, -15, -90, -90, -90, -90],
        [ -90, -65, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -65, -90, -90, -15, -90, -90, -90, -90],
        [ -90, -65, -90, -90, -90, -90, -40, -90, -90],
        [ -90, -90, -90, -90, -90, -40, -40, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90]],
        # Upside down
        [[ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -65, -65, -65, -65, -65, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -90, -15, -90, -15, -90, -90, -90],
        [ -90, -40, -90, -90, -90, -90, -90, -40, -90],
        [ -90, -40, -40, -90, -90, -90, -40, -40, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90]],
        # Counterclockwise turn
        [[ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -40, -40, -90, -90, -90, -90, -90],
        [ -90, -90, -40, -90, -90, -90, -90, -65, -90],
        [ -90, -90, -90, -90, -15, -90, -90, -65, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -65, -90],
        [ -90, -90, -90, -90, -15, -90, -90, -65, -90],
        [ -90, -90, -40, -90, -90, -90, -90, -65, -90],
        [ -90, -90, -40, -40, -90, -90, -90, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90]],
        # New face
        [[ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -40, -40, -90, -90, -90, -40, -40, -90],
        [ -90, -40, -90, -90, -90, -90, -90, -40, -90],
        [ -90, -90, -90, -90, -15, -90, -90, -90, -90],
        [ -90, -90, -90, -90, -15, -90, -90, -90, -90],
        [ -90, -90, -65, -90, -90, -90, -65, -90, -90],
        [ -90, -90, -90, -65, -65, -65, -90, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90]],
        # New face 2
        [[ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -40, -40, -40, -90, -40, -40, -40, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -90, -15, -90, -15, -90, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
        [ -90, -90, -90, -90, -65, -65, -90, -90, -90],
        [ -90, -90, -90, -65, -65, -65, -90, -90, -90],
        [ -90, -90, -90, -90, -90, -90, -90, -90, -90]],
    ]
    new_goals = [
        # Clockwise turn
        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 3, 3, 1, 1],
        [ 1, 2, 1, 1, 1, 1, 3, 1, 1],
        [ 1, 2, 1, 1, 4, 1, 1, 1, 1],
        [ 1, 2, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 2, 1, 1, 4, 1, 1, 1, 1],
        [ 1, 2, 1, 1, 1, 1, 3, 1, 1],
        [ 1, 1, 1, 1, 1, 3, 3, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        # Upside down
        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 2, 2, 2, 2, 2, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 4, 1, 4, 1, 1, 1],
        [ 1, 3, 1, 1, 1, 1, 1, 3, 1],
        [ 1, 3, 3, 1, 1, 1, 3, 3, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        # Counterclockwise turn
        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 3, 3, 1, 1, 1, 1, 1],
        [ 1, 1, 3, 1, 1, 1, 1, 2, 1],
        [ 1, 1, 1, 1, 4, 1, 1, 2, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 2, 1],
        [ 1, 1, 1, 1, 4, 1, 1, 2, 1],
        [ 1, 1, 3, 1, 1, 1, 1, 2, 1],
        [ 1, 1, 3, 3, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        # New face
        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 3, 3, 1, 1, 1, 3, 3, 1],
        [ 1, 3, 1, 1, 1, 1, 1, 3, 1],
        [ 1, 1, 1, 1, 4, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 4, 1, 1, 1, 1],
        [ 1, 1, 2, 1, 1, 1, 2, 1, 1],
        [ 1, 1, 1, 2, 2, 2, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        # New face 2
        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 3, 3, 3, 1, 3, 3, 3, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 4, 1, 4, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 2, 2, 1, 1, 1],
        [ 1, 1, 1, 2, 2, 2, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
    ]
    options = [(stim, goal) for stim,goal in zip(new_stims, new_goals)]
    return options

for file_name in file_names:
    current_dir = f"/cluster/tufts/levinlab/shansa01/SFC/sfc_results/{file_name}"
    print(current_dir, flush=True)

    # Loading exp
    with open(f"{current_dir}/exp.pickle", "rb") as fp:
        exp = pickle.load(fp)

    # Loading Net
    net = NEAT.NeuralNetwork()
    net.Load(f"{current_dir}/winner_net.txt")
    net.Flush()

    #TESTS 3/14/23
    def test_eval_individual(net, exp):
        fit = 0
        for i in range(5):
            model = Multicellularity_model(net = net, exp = exp)
            model.verbose = False
            run = model.run_model(fitness_evaluation=True)
            fit += run
        fit/=5
        return fit
    
    def new_eval(net,exp, baseline):
        fit200 = 0
        wins=0
        for i in range(5):
            model = Multicellularity_model(net = net, exp = exp)
            model.verbose = False
            run = model.run_model(fitness_evaluation=True)
            run200 = model.run_model(fitness_evaluation=True)
            print(f"Fit at 200: {int(run200)}, Baseline: {int(baseline)}")
            if run200 >= (baseline- 5):
                return True, fit200
        print("---------")
        return False, fit200

    ends = getOptions()
    baseline=0
    for x in range(3):
        b = test_eval_individual(net, exp)
        baseline+=b
    baseline = int(baseline/3)

    extra_generalizable = 0 
    for s in ends:
        new_stim = s[0][::-1]
        new_goal = s[1][::-1]
        exp.preset =    [       
            ("preset_bioelectric", [100, new_stim]),
            ("preset_goal", [100, new_goal]),
            ("preset_reset_energy", [100, exp.energy]),
        ]
        tot=0
        bol, f200 = new_eval(net, exp, baseline)
        if bol:
            extra_generalizable+=1
        print("~~~~~~~~~~~")
    print(f"{file_name} has a EXTRA generalizability score of {extra_generalizable} out of 5!", flush=True)
