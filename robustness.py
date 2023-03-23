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
# file_names='''
# 2023/Mar/04/14:41:26_95
# 2023/Mar/05/08:28:45_86
# 2023/Mar/05/03:07:34_85
# 2023/Mar/06/04:44:23_85
# 2023/Mar/05/13:35:02_85
# 2023/Mar/04/11:54:08_93
# 2023/Mar/05/00:50:47_94
# 2023/Mar/05/10:53:55_91
# 2023/Mar/05/15:48:48_93
# 2023/Mar/05/23:24:29_92
# 2023/Mar/12/06:54:00_88
# 2023/Mar/11/20:15:26_87
# 2023/Mar/11/18:16:09_87
# 2023/Mar/11/16:23:25_85
# 2023/Mar/12/05:04:05_84
# 2023/Mar/12/22:20:34_83
# 2023/Mar/12/07:49:36_74
# 2023/Mar/12/13:56:17_71
# 2023/Mar/12/05:47:11_64
# 2023/Mar/12/11:54:16_60
# 2023/Mar/04/23:26:04_98
# 2023/Mar/04/23:40:15_98
# 2023/Mar/05/16:12:36_98
# 2023/Mar/05/07:47:01_99
# 2023/Mar/05/15:09:07_98
# 2023/Mar/11/09:18:33_97
# 2023/Mar/14/02:02:25_97
# 2023/Mar/14/00:14:29_96
# 2023/Mar/13/22:25:14_96
# 2023/Mar/13/18:53:23_96
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

for f in range(len(file_names)):
    file_name = file_names[f]
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

    robustness=0 
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
            robustness+=1
        print(val, baseline, flush=True)
        # print(max(vals), baseline)
        # print(vals)
    print(f"{file_name} has a robustness score of {robustness} out of 20!", flush=True)
    all_scores.append(robustness)

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
