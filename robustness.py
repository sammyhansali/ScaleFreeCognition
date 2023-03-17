import pickle
import MultiNEAT as NEAT
import os
from analysis import sim
from multicellularity.model_for_analysis import Multicellularity_model
from random_faces import RandomFaces
import sys


# Getting absolute path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_name = sys.argv[1]
file_names='''
2023/Mar/04/14:41:26_95
2023/Mar/05/08:28:45_86
2023/Mar/05/03:07:34_85
2023/Mar/06/04:44:23_85
2023/Mar/05/13:35:02_85
2023/Mar/04/11:54:08_93
2023/Mar/05/00:50:47_94
2023/Mar/05/10:53:55_91
2023/Mar/05/15:48:48_93
2023/Mar/05/23:24:29_92
2023/Mar/12/06:54:00_88
2023/Mar/11/20:15:26_87
2023/Mar/11/18:16:09_87
2023/Mar/11/16:23:25_85
2023/Mar/12/05:04:05_84
2023/Mar/12/22:20:34_83
2023/Mar/12/07:49:36_74
2023/Mar/12/13:56:17_71
2023/Mar/12/05:47:11_64
2023/Mar/12/11:54:16_60
2023/Mar/04/23:26:04_98
2023/Mar/04/23:40:15_98
2023/Mar/05/16:12:36_98
2023/Mar/05/07:47:01_99
2023/Mar/05/15:09:07_98
2023/Mar/11/09:18:33_97
2023/Mar/14/02:02:25_97
2023/Mar/14/00:14:29_96
2023/Mar/13/22:25:14_96
2023/Mar/13/18:53:23_96
'''.split()

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
            # print(f"Run {i+1}: {round(run,1)}")
        fit/=5
        # print(f"Expected fitness (analysis.py): {fit}")
        return fit

    starts = RandomFaces().options
    baseline=0
    for x in range(3):
        baseline += test_eval_individual(net, exp)
    baseline = int(baseline/3)

    generalizable=0 
    for s in starts:
        exp.start = s[::-1]
        # vals = []
        # vals.append(test_eval_individual(net, exp))
        val=0
        for x in range(3):
            val += test_eval_individual(net, exp)
        val= int(val/3)
        # if max(vals) >= (baseline - 5):
        if val >= (baseline - 5):
            generalizable+=1
        print(val, baseline, flush=True)
        # print(max(vals), baseline)
        # print(vals)
    print(f"{file_name} has a robustness score of {generalizable} out of 20!", flush=True)
    # # Simulation
    # sim(exp, net)