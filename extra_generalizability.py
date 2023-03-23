import pickle
import MultiNEAT as NEAT
import os
from analysis import sim
from multicellularity.model_for_analysis import Multicellularity_model
from random_faces import RandomFaces
import sys


### JUST HAVE TO BE ABLE TO EXTRA GENERALIZE ONCE - SINCE THERE IS TOO MUCH ERROR/VARIANCE

# file_names='''
# 2023/Mar/13/22:25:14_96
# '''.split()
file_names = '''
2023/Mar/20/19:58:46_98
2023/Mar/21/03:40:20_97
2023/Mar/20/16:10:47_97
2023/Mar/20/17:26:42_96
2023/Mar/21/07:18:43_95
2023/Mar/20/11:42:28_97
2023/Mar/20/21:57:16_96
2023/Mar/20/16:46:15_91
2023/Mar/20/14:17:42_88
2023/Mar/21/04:22:24_88
2023/Mar/20/19:35:58_91
2023/Mar/20/08:00:36_88
2023/Mar/20/23:56:08_87
2023/Mar/21/05:31:51_86
2023/Mar/21/01:19:04_85
2023/Mar/20/09:33:35_95
2023/Mar/21/01:28:58_90
2023/Mar/21/09:57:31_89
2023/Mar/20/12:49:47_88
2023/Mar/21/06:26:29_83
2023/Mar/21/01:24:39_93
2023/Mar/20/17:19:47_88
2023/Mar/20/15:27:19_88
2023/Mar/20/23:24:22_87
2023/Mar/21/15:23:29_85
'''.split()

all_scores = []

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
    all_scores.append(extra_generalizable)

import statistics
mode_1_1 = all_scores[:5]
mode_1_2 = all_scores[5:10]
mode_1_3 = all_scores[10:15]
mode_1_4 = all_scores[15:20]
mode_1_5 = all_scores[20:25]
modes=[mode_1_1, mode_1_2, mode_1_3, mode_1_4, mode_1_5]
for mode in modes:
    avg = statistics.mean(mode)
    st = statistics.stdev(mode)
    mi = min(mode)
    ma = max(mode)
    print(f"This mode has an average robust score of {avg}, with standard deviation of {st}, min of {mi}, and max of {ma}")