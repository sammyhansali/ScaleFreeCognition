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
import json

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
    global file
    file = myString.replace(' ','_')
    os.makedirs("SCS_Results/" + file, exist_ok=True)
    
    # Saving files and folders
    shutil.copyfile("analysis.py",  "SCS_Results/" + file + "/" + "analysis.py")
    shutil.copyfile("run_analysis.py",  "SCS_Results/" + file + "/" + "run_analysis.py")
    shutil.copyfile("random_faces.py",  "SCS_Results/" + file + "/" + "random_faces.py")
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
    best_genome = []
    best_fitness = -20000
    best_ID = -1

    for generation in range(exp.nb_gens):
        gen_time = time.time()

        # Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)
        ff = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, ff)]


        gen_best_genome = pop.Species[0].GetLeader()   
        gen_best_fitness = gen_best_genome.GetFitness()
        gen_best_ID = gen_best_genome.GetID()

        if (gen_best_fitness > best_fitness):
            # best_genome = gen_best_genome
            best_genome.append(gen_best_genome)
            best_fitness = gen_best_fitness
            best_ID = gen_best_ID

        # Advance to the next generation
        pop.Epoch()

        # Print generation's statistics        
        gen_elapsed_time = time.time() - gen_time
        output = f"""
        *****************************************************
        GEN: {generation}

            Generation best fitness:    {round(gen_best_fitness,1)},    ID: {gen_best_ID}
            Generation elapsed time:    {round(gen_elapsed_time,1)}
            Trial best fitness so far:  {round(best_fitness,1)},    ID: {best_ID}
        *****************************************************
        """
        print(textwrap.dedent(output))

        solution_found = (gen_best_fitness >= exp.max_fitness) 
        if solution_found:
            break

    ## Print trial statistics
    elapsed_time = time.time() - start_time
    file_name = "_".join([file,str(int(best_fitness))]).replace(":", "\:")
    output = f"""
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    EVOLUTION DONE

        Trial best fitness:     {round(best_fitness,1)},    ID: {best_ID}
        Trial elapsed time:     {round(elapsed_time,1)}
        File:                   {file_name}
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    print(textwrap.dedent(output))

    # Match expectations?
    bst = best_genome[-1]

    # Saving score
    with open("score.txt","w+") as score_file:
        score_file.write("\nBest ever fitness: %f, genome ID: %d" % (best_fitness, best_ID))
        score_file.write("\nTrial elapsed time: %.3f sec" % (elapsed_time))
        # score_file.write("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))
    os.replace("score.txt", "SCS_Results/" + file + "/" + "score.txt")

    # Saving genome
    bst.Save("best_genome.txt")
    os.replace("best_genome.txt", "SCS_Results/" + file + "/" + "best_genome.txt")
    # Pickling method (may not work right)
    with open("best_genome.pickle", 'wb') as f:
        pickle.dump(bst, f, pickle.HIGHEST_PROTOCOL)
    os.replace("best_genome.pickle", "SCS_Results/" + file + "/" + "best_genome.pickle")

    # Visualize best network's Genome
    winner_net_CPPN = NEAT.NeuralNetwork()
    bst.BuildPhenotype(winner_net_CPPN)

    # Visualize best network's Phenotype
    winner_net = NEAT.NeuralNetwork()
    bst.BuildESHyperNEATPhenotype(winner_net, exp.substrate, exp.params)
    winner_net.Save("SCS_Results/" + file + "/winner_net.txt")

    #save_params
    # params_file = os.path.join(".", "params.txt")
    # with open(params_file, 'wb') as p1:
        # pickle.dump(exp.params, p1)
    np.save("SCS_Results/" + file + "/" +  "params.txt", exp.params)
    # os.replace(params_file, "SCS_Results/" + file + "/" +  "params.txt")

    #save_substrate 
    # substrate_file = os.path.join(".", "substrate.txt")
    # with open(substrate_file, 'wb') as s1:
        # pickle.dump(exp.substrate, s1)
    np.save("SCS_Results/" + file + "/" + "substrate.txt", exp.substrate)
    # os.replace(substrate_file, "SCS_Results/" + file + "/" + "substrate.txt")

    # Pickle the experiment (needed for analysis)
    exp_file = os.path.join(".", "exp.pickle")
    with open(exp_file, 'wb') as f:
        pickle.dump(exp, f, pickle.HIGHEST_PROTOCOL)
    os.replace(exp_file, "SCS_Results/" + file + "/" +  "exp.pickle")
    ##### Below is experimental

    ## Here the experiment is normal, not the unpickled one
    fp = "SCS_Results/" + file + "/" + "winner_net.txt"
    net = NEAT.NeuralNetwork()
    net.Load(fp)
    fit=0
    for i in range(5):
        run = test_eval_individual(net, exp)
        print(run)
        fit+=run
    print("Expected", best_fitness)
    print("Actual", str(round(fit/5,1)))

    os.rename('SCS_Results/' + file,'SCS_Results/' + file +'_'+ str(int(best_fitness))) # used to be "Results/" for both

    from analysis import sim
    if exp.simulate == True:
        sim(exp, winner_net)

def eval_individual(genome):
    
    """
    Evaluate fitness of the individual CPPN genome by creating
    the substrate with topology based on the CPPN output.
    Arguments:
        genome:         The CPPN genome
        substrate:      The substrate to build control ANN
        params:         The ES-HyperNEAT hyper-parameters
    Returns:
        fitness_individual The fitness of the individual
    """
    global g_exp
    
    net = NEAT.NeuralNetwork()
    genome.BuildESHyperNEATPhenotype(net, g_exp.substrate, g_exp.params)
    net.Flush()

    # random_start == true means that each generation should be trained on a different random face, to get a robust NN.
    if g_exp.random_start==True:
        g_exp.start = RandomFaces().get_random_face()

    fit = 0
    for i in range(5):
        # model = Multicellularity_model(net = net, exp = g_exp)
        model = Multicellularity_model(
            net = net, 
            exp = g_exp,
        )
        model.verbose = False
        trial = model.run_model(fitness_evaluation=True)
        fit += trial
    fit /= 5

    if fit > 95:
        global file
        net.Save("SCS_Results/" + file + "/winner_net_"+ str(round(fit,1))+".txt")
    
    return fit

def test_eval_individual(net, exp):
    
    fit = 0
    for i in range(5):
        model = Multicellularity_model(net = net, exp = exp)
        model.verbose = False
        run = model.run_model(fitness_evaluation=True)
        fit += run
    fit/=5
    return fit