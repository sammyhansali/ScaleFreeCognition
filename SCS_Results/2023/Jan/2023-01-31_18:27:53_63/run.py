from multicellularity.model_for_analysis import Multicellularity_model
try:
   import cPickle as pickle
except:
   import pickle
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from multicellularity.agents import Cell
from random_faces import RandomFaces

import os
import sys
import time
import random
import subprocess as comm
import cv2
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
from MultiNEAT import GetGenomeList, ZipFitness
from MultiNEAT import  EvaluateGenomeList_Parallel, EvaluateGenomeList_Serial
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import datetime
import shutil
import multiprocessing
import textwrap

cpu_number = multiprocessing.cpu_count()
print(cpu_number)


file="" # Make file a global variable so eval_individual can use it.
g_exp=None # global experiment

def run_experiment(exp):
    """
        exp: the experiment parameters
    """
    # Updating global experiment
    global g_exp
    g_exp = exp

    # Result file
    myDatetime = datetime.datetime.now()
    myString = myDatetime.strftime('%Y-%m-%d %H:%M:%S')
    file = myString.replace(' ','_')
    os.makedirs("SCS_Results/" + file, exist_ok=True)
    
    # Saving files and folders
    shutil.copyfile("analysis.py",  "SCS_Results/" + file + "/" + "analysis.py")
    shutil.copyfile("run.py",  "SCS_Results/" + file + "/" + "run.py")
    shutil.copyfile("experiment.py",  "SCS_Results/" + file + "/" + "experiment.py")
    shutil.copytree('./multicellularity', "SCS_Results/" + file + "/" + "multicellularity")
    shutil.copytree('./Experiments', "SCS_Results/" + file + "/" + "Experiments")

    # Save random seed and exp.params
    seed = int(time.time()) #1660341957#
    np.save("SCS_Results/" + file + "/seed", seed)
    # exp.params.Save("SCS_Results/" + file + "/multiNEAT_params.txt")

    genome = NEAT.Genome(0,
                    exp.substrate.GetMinCPPNInputs(),
                    2, # hidden units
                    exp.substrate.GetMinCPPNOutputs(),
                    False,
                    NEAT.ActivationFunction.TANH,
                    NEAT.ActivationFunction.SIGNED_GAUSS,
                    1, # hidden layers seed
                    exp.params, 
                    1)  # one hidden layer
    
    pop = NEAT.Population(genome, exp.params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    # Run for up to N generations.
    start_time = time.time()
    best_genome = None
    best_fitness = -20000
    best_ID = -1

    for generation in range(exp.nb_gens):
        gen_time = time.time()

        # Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)
        # fitnesses = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)
        # [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]
        fitnesses = yolo(genome_list, eval_individual, display=False, cores=cpu_number)
        if generation == 1:
            print(fitnesses)
        [genome.SetFitness(fitness[1]) for genome, fitness in zip(genome_list, fitnesses)]

        gen_best_genome = pop.Species[0].GetLeader()   
        gen_best_fitness = gen_best_genome.GetFitness()
        gen_best_ID = gen_best_genome.GetID()

        solution_found = (gen_best_fitness >= exp.max_fitness) 
        if (gen_best_fitness > best_fitness):
            best_genome = gen_best_genome
            best_fitness = gen_best_fitness
            best_ID = gen_best_genome.GetID()

            if solution_found:
                print(f'Solution found at generation: {generation}, best fitness: {round(best_fitness, 1)}')
                break
        
        # Advance to the next generation
        pop.Epoch()

        # Print generation's statistics        
        gen_elapsed_time = time.time() - gen_time
        output = f"""
        *******************************************
        GEN: {generation}

            Generation best fitness:    {round(gen_best_fitness,1)},    ID: {gen_best_ID}
            Generation elapsed time:    {round(gen_elapsed_time,1)}
            Trial best fitness so far:  {round(best_fitness,1)},    ID: {best_ID}
        *******************************************
        """
        print(textwrap.dedent(output))

    ## Print trial statistics
    elapsed_time = time.time() - start_time
    file_name = "_".join([file,str(int(best_fitness))]).replace(":", "\:")
    output = f"""
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    EVOLUTION DONE

        Trial best fitness:     {round(best_fitness,1)},    ID: {best_ID}
        Trial elapsed time:     {round(elapsed_time,1)}
        File:                   {file_name}
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    print(textwrap.dedent(output))


    # Saving score
    with open("score.txt","w+") as score_file:
        score_file.write("\nBest ever fitness: %f, genome ID: %d" % (best_fitness, best_ID))
        score_file.write("\nTrial elapsed time: %.3f sec" % (elapsed_time))
        # score_file.write("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))
    os.replace("score.txt", "SCS_Results/" + file + "/" + "score.txt")

    # Visualize best network's Genome
    # winner_net_CPPN = NEAT.NeuralNetwork()
    # best_genome.BuildPhenotype(winner_net_CPPN)
    # print("\nCPPN nodes: %d, connections: %d" % (len(winner_net_CPPN.neurons), len(winner_net_CPPN.connections)))

    # Visualize best network's Phenotype
    winner_net = NEAT.NeuralNetwork()
    best_genome.BuildESHyperNEATPhenotype(winner_net, exp.substrate, exp.params)
    winner_net.Save("SCS_Results/" + file + "/winner_net.txt")
    # print("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))
    
    # Pickle the experiment (needed for analysis)
    exp_file = os.path.join(".", "exp.pickle")
    with open(exp_file, 'wb') as f:
        pickle.dump(exp, f)
    os.replace(exp_file, "SCS_Results/" + file + "/" +  "exp.pickle")

    # Run analysis for visualization
    import subprocess
    subprocess.Popen("python3 analysis.py", cwd="SCS_Results/" + file + "/", shell=True) # used to be "Results/"
    os.rename('SCS_Results/' + file,'SCS_Results/' + file +'_'+ str(int(best_fitness))) # used to be "Results/" for both



def eval_individual(genome):
    
    """
    Evaluate fitness of the individual CPPN genome by creating
    the substrate with topology based on the CPPN output.
    Arguments:
        genome:         The CPPN genome
        substrate:      The substrate to build control ANN
        params:         The ES-HyperNEAT hyper-parameters
    Returns:
        fitness_indiidual The fitness of the individual
    """
    
    net = NEAT.NeuralNetwork()
    # return genome # Test
    genome.BuildESHyperNEATPhenotype(net, g_exp.substrate, g_exp.params)
    depth = g_exp.params.MaxDepth
    net.Flush()

    # print(g_exp.start)
    # random_start == true means that each generation should be trained on a different random face, to get a robust NN.
    if g_exp.random_start==True:
        g_exp.start = RandomFaces().get_random_face()

    model = Multicellularity_model(
        net = net, 
        exp = g_exp,
    )
    model.verbose = False
    fit1 = model.run_model()

    # model2 = Multicellularity_model(
    #         net = net, 
    #         exp = g_exp,
    # )
    # model2.verbose = False
    # fit2 = model2.run_model()

    # fitness_test = (fit1 + fit2)/2
    fitness_test = fit1
    return fitness_test



def yolo(genome_list, evaluator, cores=8, display=True, ipython_client=None):
    fitnesses = []
    curtime = time.time()

    if ipython_client is None or not ipython_installed:
        with ProcessPoolExecutor(max_workers=cores) as executor:
            for i, fitness in enumerate(executor.map(evaluator, genome_list)):
                fitnesses.append((i, fitness))

                if display:
                    if ipython_installed: clear_output(wait=True)
                    print('Individuals: (%s/%s) Fitness: %3.4f' % (i, len(genome_list), fitness))
    else:
        if type(ipython_client) == Client:
            lbview = ipython_client.load_balanced_view()
            amr = lbview.map(evaluator, genome_list, ordered=True, block=False)
            for i, fitness in enumerate(amr):
                if display:
                    if ipython_installed: clear_output(wait=True)
                    print('Individual:', i, 'Fitness:', fitness)
                fitnesses.append((i, fitness))
        else:
            raise ValueError('Please provide valid IPython.parallel Client() as ipython_client')

    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %3.4f' % elapsed)

    return sorted(fitnesses, key=lambda x: x[0])