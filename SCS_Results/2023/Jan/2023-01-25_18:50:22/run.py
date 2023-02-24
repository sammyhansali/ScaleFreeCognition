## 4 directions


from multicellularity.model_for_analysis import Multicellularity_model
try:
   import cPickle as pickle
except:
   import pickle
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from multicellularity.agents import Cell

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
import multicellularity.visualize as visualize
import sys
import datetime
import shutil
import multiprocessing

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
    
    # Save general params
    Result_file = "SCS_Results/" + file + "/"+ "general_params.txt"
    
    with open("general_params.txt","w+") as general_params_file:
        print("depth: %s  \n\
height: %s  \n\
width: %s  \n\
energy: %s \n\
nb_output_molecules: %s  \n\
history_length: %s  \n\
multiple: %s  \n\
e_penalty: %s  \n\
step_count: %s" % (exp.depth, exp.height, exp.width, 
        exp.energy, exp.nb_output_molecules, exp.history_length, exp.multiple, exp.e_penalty, exp.step_count), 
        file=general_params_file)
    os.replace("general_params.txt", Result_file)
        
    # Saving files and folders
    shutil.copyfile("analysis.py",  "SCS_Results/" + file + "/" + "analysis.py")
    shutil.copyfile("run.py",  "SCS_Results/" + file + "/" + "run.py")
    shutil.copyfile("experiment.py",  "SCS_Results/" + file + "/" + "experiment.py")
    shutil.copytree('./multicellularity', "SCS_Results/" + file + "/" + "multicellularity")
    shutil.copytree('./Experiments', "SCS_Results/" + file + "/" + "Experiments")

    # Save random seed and exp.params
    seed = int(time.time()) #1660341957#
    np.save("SCS_Results/" + file + "/seed", seed)
    exp.params.Save("SCS_Results/" + file + "/multiNEAT_params.txt")

    # Matrices
    np.savetxt("SCS_Results/" + file + "/start_matrix.txt", exp.start)
    np.savetxt("SCS_Results/" + file + "/goal_matrix.txt", exp.goal)
    # np.savetxt("SCS_Results/" + file + "/preset.txt", exp.preset)
    with open("SCS_Results/" + file + "/preset.txt", "wb") as fp:   #Pickling
        pickle.dump(exp.preset, fp)
    if exp.bioelectric_stimulus is not None:
        np.savetxt("SCS_Results/" + file + "/bioelectric_stimulus_matrix.txt", exp.bioelectric_stimulus)
    else:
        np.savetxt("SCS_Results/" + file + "/bioelectric_stimulus_matrix.txt", np.zeros(1)) # Blank

    # exp.start
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
    # best_genome_ser = None
    best_ever_goal_fitness = -20000
    best_id = -1
    solution_found = False
    plot_best_fitness=[]
    
    for generation in range(exp.nb_gens):
        gen_time = time.time()

        # Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)

        #test
        # fitnesses = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)
        # [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]
        fitnesses = EvaluateGenomeList_Serial(genome_list, eval_individual, display=False, cores=cpu_number)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        # gf_pairs = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)
        # new_genome_list = [genome for genome, fitness in gf_pairs]
        # # print(len(genome_list) == len(new_genome_list))
        # print(genome_list)
        # print(new_genome_list)
        # [genome.SetFitness(fitness) for genome, fitness in gf_pairs]

        # genomes = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)

        
        # solution_found = max(fitnesses) >= exp.max_fitness
        # Store the best genome
        mm = 0
        mm_genome = None
        for genome in genome_list:
            if genome.GetFitness() >= mm:
                mm = genome.GetFitness()
                mm_genome = genome
            print("fit ", genome.GetFitness())
        print("mm", mm)
        gen_best_genome = mm_genome
        print(gen_best_genome.GetFitness())
        # print(dir(pop))
        # print(dir(pop.Species[0]))
        # print(dir(genome_list[0]))
        gen_best_genome = pop.Species[0].GetLeader()   
        print(gen_best_genome.GetFitness())
        gen_best_ID = gen_best_genome.GetID()
        if solution_found or best_ever_goal_fitness < max(fitnesses):
            print("hi")
            best_genome = gen_best_genome
            # best_genome_ser = pickle.dumps(gen_best_genome) # dump to pickle to freeze the genome state
            best_ever_goal_fitness = max(fitnesses)
            print(best_ever_goal_fitness)
            print(best_genome.GetFitness())
            best_id = gen_best_genome.GetID()

        if solution_found:
            print('Solution found at generation: %d, best fitness: %f, species count: %d' % (generation, max(fitnesses), len(pop.Species)))
            break
        
        # advance to the next generation
        pop.Epoch()
        plot_best_fitness.append(max(fitnesses))

        # print statistics        
        gen_elapsed_time = time.time() - gen_time
        print("")
        print('*******************************************')
        print("Generation: %d" % generation)
        print("Best fitness: %f, genome ID: %d" % (max(fitnesses), gen_best_ID))
        print("Species count: %d" % len(pop.Species))
        print("Generation elapsed time: %.3f sec" % (gen_elapsed_time))
        print("Best fitness ever: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
        print('*******************************************')
        print("")

    print(best_genome.GetFitness())
    elapsed_time = time.time() - start_time
    # Print experiment statistics
    print("\nBest ever fitness: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
    print("\nTrial elapsed time: %.3f sec" % (elapsed_time))
    print("Random seed:", seed)




    # Visualize best network's Genome
    winner_net_CPPN = NEAT.NeuralNetwork()
    # gen_best_genome.BuildPhenotype(winner_net_CPPN)
    best_genome.BuildPhenotype(winner_net_CPPN)
    print("\nCPPN nodes: %d, connections: %d" % (len(winner_net_CPPN.neurons), len(winner_net_CPPN.connections)))

    # Visualize best network's phenotype
    winner_net = NEAT.NeuralNetwork()
    # gen_best_genome.BuildESHyperNEATPhenotype(winner_net, exp.substrate, exp.params)
    best_genome.BuildESHyperNEATPhenotype(winner_net, exp.substrate, exp.params)
    winner_net.Save("SCS_Results/" + file + "/winner_net.txt")
    print("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))




    # Write best genome instance to file
    # best_genome_ser = pickle.dumps(gen_best_genome)
    # best_genome = pickle.loads(best_genome_ser)
    best_genome_file = os.path.join(".", "best_genome.pickle")
    np.save('plot_best_fitness', plot_best_fitness)
    with open(best_genome_file, 'wb') as f:
        pickle.dump(best_genome, f, pickle.HIGHEST_PROTOCOL)
    os.replace(best_genome_file, "SCS_Results/" + file + "/" + "best_genome.pickle")

    # Saving score
    with open("score.txt","w+") as score_file:
        score_file.write("\nBest ever fitness: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
        score_file.write("\nTrial elapsed time: %.3f sec" % (elapsed_time))
        score_file.write("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))
    score_file.close() 
    os.replace("score.txt", "SCS_Results/" + file + "/" + "score.txt")
    
    #save_params
    params_file = os.path.join(".", "params.pickle")
    with open(params_file, 'wb') as param_file1:
        pickle.dump(exp.params, param_file1)
    param_file1.close() 
    os.replace(params_file, "SCS_Results/" + file + "/" +  "params.pickle")

    # save the entire experiment instance, and experiment class (for pickle)
    #test
    # print(exp.ANN_inputs)
    # print(exp.ANN_outputs)
    exp_file = os.path.join(".", "exp.pickle")
    with open(exp_file, 'wb') as f:
        pickle.dump(exp, f)
    os.replace(exp_file, "SCS_Results/" + file + "/" +  "exp.pickle")
    # shutil.copyfile("experiment.py",  "SCS_Results/" + file + "/" + "experiment.py")

    # ANN inputs
    inp_file = os.path.join(".", "ANN_inputs.pickle")
    with open(inp_file, 'wb') as f:
        pickle.dump(exp.ANN_inputs, f)
    os.replace(inp_file, "SCS_Results/" + file + "/" +  "ANN_inputs.pickle")
    # ANN outputs
    out_file = os.path.join(".", "ANN_outputs.pickle")
    with open(out_file, 'wb') as f:
        pickle.dump(exp.ANN_outputs, f)
    os.replace(out_file, "SCS_Results/" + file + "/" +  "ANN_outputs.pickle")
    
    #save_substrate 
    substrate_file = os.path.join(".", "substrate.pickle")
    with open(substrate_file, 'wb') as substrate_file1:
        pickle.dump(exp.substrate, substrate_file1)
    substrate_file1.close() 
    os.replace(substrate_file, "SCS_Results/" + file + "/" + "substrate.pickle")

    print('*******************************************')
    print("EVOLUTION DONE")
    print('*******************************************')
    print("File name:" + "\t" +  "_".join([file,str(int(best_ever_goal_fitness))]).replace(":", "\:") )
    
    # Do the analysis...
    import subprocess
    subprocess.Popen("python3 analysis.py", cwd="SCS_Results/" + file + "/", shell=True) # used to be "Results/"
    os.rename('SCS_Results/' + file,'SCS_Results/' + file +'_'+ str(int(best_ever_goal_fitness))) # used to be "Results/" for both



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
    genome.BuildESHyperNEATPhenotype(net, g_exp.substrate, g_exp.params)
    depth = g_exp.params.MaxDepth
    net.Flush()

    start_mat = [[]]
    # random_start == true means that each generation should be trained on a different random face, to get a robust NN.
    if g_exp.random_start:
        options =   [   # 0 (Normal)
                        [[ 3, 3, 1, 1, 1, 1, 1, 1, 1],
                        [ 3, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 2, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 2, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 2, 2, 2, 2, 2, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 3, 3],
                        [ 1, 1, 1, 1, 1, 1, 1, 3, 1]],
                        # 1
                        [[ 1, 1, 2, 2, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 3, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 3, 3, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 2, 2, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 2, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 2, 3, 3, 3],
                        [ 1, 1, 1, 1, 1, 2, 1, 1, 1]],
                        # 2
                        [[ 1, 1, 2, 2, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 2, 2, 2, 1],
                        [ 1, 1, 1, 3, 1, 2, 1, 1, 1],
                        [ 1, 1, 3, 1, 1, 2, 1, 1, 1],
                        [ 1, 1, 3, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 3, 3, 1, 1],
                        [ 1, 1, 1, 1, 1, 3, 1, 1, 1]],
                        # 3
                        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 2, 1, 1, 1],
                        [ 1, 1, 1, 1, 2, 1, 1, 2, 1],
                        [ 1, 2, 2, 2, 1, 1, 1, 1, 2],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 3, 3, 3, 3, 1, 1, 1, 1],
                        [ 1, 3, 1, 3, 1, 1, 1, 1, 1]],
                        # 4
                        [[ 1, 1, 1, 2, 2, 2, 1, 1, 1],
                        [ 1, 1, 1, 2, 1, 1, 2, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 3, 3],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 3],
                        [ 1, 1, 1, 1, 1, 3, 1, 1, 1],
                        [ 1, 1, 1, 1, 3, 3, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 2, 1, 2, 1, 1, 1, 1]],
                        # 5
                        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 3],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 3],
                        [ 2, 2, 1, 1, 1, 1, 1, 1, 3],
                        [ 1, 1, 2, 2, 2, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 3],
                        [ 1, 1, 1, 1, 1, 1, 2, 1, 3],
                        [ 1, 1, 1, 1, 1, 1, 2, 1, 3],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        # 6
                        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 3, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 3, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 3, 1, 1, 1, 1, 1, 2, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 2, 1],
                        [ 1, 3, 1, 3, 1, 1, 1, 1, 2],
                        [ 1, 1, 3, 1, 1, 1, 2, 1, 2],
                        [ 1, 1, 1, 1, 1, 1, 2, 1, 2],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        # 7
                        [[ 1, 1, 3, 3, 3, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 2, 2, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 2, 2, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 2, 1, 1, 1],
                        [ 1, 2, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 2, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 3, 3, 3, 1, 1, 1]],
                        # 8
                        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 3, 1, 3, 1, 1],
                        [ 1, 1, 1, 1, 1, 3, 1, 1, 1],
                        [ 1, 1, 2, 1, 2, 1, 2, 1, 1],
                        [ 1, 2, 1, 2, 1, 2, 1, 2, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 3, 1, 1, 1],
                        [ 1, 1, 1, 1, 3, 1, 3, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        # 9
                        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 2, 1, 2, 1, 1, 1],
                        [ 1, 1, 2, 1, 2, 1, 2, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 2, 1, 1, 1, 1],
                        [ 1, 1, 3, 1, 2, 1, 1, 1, 1],
                        [ 1, 1, 3, 3, 1, 1, 3, 1, 1],
                        [ 1, 1, 1, 1, 3, 3, 1, 1, 1]],
                        # 10
                        [[ 1, 2, 1, 1, 1, 1, 1, 1, 1],
                        [ 2, 1, 1, 1, 1, 1, 1, 3, 3],
                        [ 1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 2, 1, 1, 1, 1],
                        [ 1, 3, 1, 1, 2, 2, 2, 2, 1],
                        [ 3, 1, 3, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        # 10
                        [[ 1, 2, 1, 1, 1, 1, 1, 1, 1],
                        [ 2, 2, 2, 2, 1, 1, 3, 3, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 3, 2, 1, 1, 1, 1],
                        [ 1, 1, 3, 1, 1, 1, 1, 2, 1],
                        [ 1, 1, 3, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        # 11
                        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 2, 1, 2, 3, 3, 1, 1, 1],
                        [ 1, 1, 1, 1, 3, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 2, 1, 1, 1, 1],
                        [ 1, 1, 3, 1, 2, 1, 1, 2, 1],
                        [ 1, 1, 3, 3, 1, 2, 2, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        # 12
                        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 2, 2, 2, 1, 1],
                        [ 1, 1, 1, 1, 3, 1, 2, 1, 1],
                        [ 3, 1, 1, 1, 1, 1, 2, 1, 1],
                        [ 1, 3, 3, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 2, 3, 1],
                        [ 1, 3, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 2, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        # 12
                        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 2, 2, 2, 1, 1, 1],
                        [ 1, 1, 2, 1, 1, 1, 2, 1, 1],
                        [ 1, 2, 1, 3, 1, 1, 1, 2, 1],
                        [ 1, 1, 3, 3, 3, 1, 1, 1, 1],
                        [ 1, 1, 1, 3, 3, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        # 13
                        [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 2, 1, 1, 1, 1, 1, 1],
                        [ 1, 2, 2, 1, 1, 1, 1, 1, 1],
                        [ 1, 2, 2, 1, 1, 1, 1, 1, 1],
                        [ 1, 2, 2, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 3, 3, 3, 1],
                        [ 1, 1, 1, 1, 1, 3, 3, 3, 1]],
                        # 14
                        [[ 1, 3, 1, 3, 1, 3, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 3, 1, 3, 1, 3, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [ 1, 2, 1, 2, 1, 2, 1, 2, 1],
                        [ 1, 1, 1, 1, 1, 1, 2, 1, 1],
                        [ 1, 1, 2, 1, 2, 1, 1, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                    ]
        start_mat = random.choice(options)
    else:
        start_mat = g_exp.start

    model = Multicellularity_model(
            net = net, 
            exp = g_exp,
            # depth = depth, # Eveything below here should not be a parameter of model... should be imported.
            # height= g_exp.height, 
            # width= g_exp.width, 
            # energy = g_exp.energy,
            # step_count = g_exp.step_count,
            # nb_output_molecules = g_exp.nb_output_molecules,
            # goal = g_exp.goal, 
            # start = start_mat,
            # bioelectric_stimulus = g_exp.bioelectric_stimulus,
            # ANN_inputs = g_exp.ANN_inputs,
            # ANN_outputs = g_exp.ANN_outputs,
            # history_length = g_exp.history_length,
            # e_penalty = g_exp.e_penalty,
            # preset = g_exp.preset,
            # multiple = g_exp.multiple,
    )
   
    model.verbose = False
    fitness_individual=0
    # print("INDIVIDUAL_START")
    fitness_test = model.run_model()
    if fitness_test>95:
        for i in range(10):
            fitness_individual += fitness_test
            fitness_test = model.run_model()
            #net.Flush()
        fitness_individual = fitness_individual/10
        net.Save("SCS_Results/" + file + "/winner_net_"+ str(fitness_individual)+".txt")
    else: 
        fitness_individual = fitness_test
        
    # print("INDIVIDUAL_DONE")
    # genome.SetFitness(fitness_individual)
    print("Genome fit indiv", genome.GetFitness())
    print("Fit indiv", fitness_individual)
    # return genome
    return (genome, fitness_individual)












