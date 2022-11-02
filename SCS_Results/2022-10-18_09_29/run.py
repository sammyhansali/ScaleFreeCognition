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
import random as rnd
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
    myString = myDatetime.strftime('%Y-%m-%d %H:%M')
    file = myString.replace(' ','_')
    
    os.makedirs("../SCS_Results/" + file, exist_ok=True)
    
    # Save general params
    Result_file = "../SCS_Results/" + file + "/"+ "general_params.txt"
    
    with open("general_params.txt","w+") as general_params_file:
        print("nb_gens: %s \n\
depth: %s  \n\
height: %s  \n\
width: %s  \n\
max_fitness: %s  \n\
energy: %s \n\
nb_gap_junctions: %s  \n\
step_count: %s" % (exp.nb_gens, exp.depth, exp.height, exp.width, 
        exp.max_fitness, exp.energy, exp.nb_gap_junctions, exp.step_count), 
        file=general_params_file)
        
    general_params_file.close() 
    os.replace("general_params.txt", Result_file)
    shutil.copyfile("analysis.py",  "../SCS_Results/" + file + "/" + "analysis.py")
    shutil.copyfile("run.py",  "../SCS_Results/" + file + "/" + "run.py")
    shutil.copytree('./multicellularity', "../SCS_Results/" + file + "/" + "multicellularity")
    

    # random seed
    seed = int(time.time()) #1660341957#
    np.save("../SCS_Results/" + file + "/seed", seed)
    exp.params.Save("../SCS_Results/" + file + "/multiNEAT_params.txt")
    
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
    best_genome_ser = None
    best_ever_goal_fitness = -20000
    best_id = -1
    solution_found = False
    plot_best_fitness=[]
    
    for generation in range(exp.nb_gens):
        
        gen_time = time.time()

        # Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)

        fitnesses = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        
        # Store the best genome
        solution_found = max(fitnesses) >= exp.max_fitness
        gen_best_genome = pop.Species[0].GetLeader()   
        gen_best_ID = gen_best_genome.GetID()
        if solution_found or best_ever_goal_fitness < max(fitnesses):
            best_genome_ser = pickle.dumps(gen_best_genome) # dump to pickle to freeze the genome state
            best_ever_goal_fitness = max(fitnesses)
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


    elapsed_time = time.time() - start_time
    best_genome = pickle.loads(best_genome_ser)
    # write best genome to the file
    best_genome_file = os.path.join(".", "best_genome.pickle")
    np.save('plot_best_fitness', plot_best_fitness)
    
    with open(best_genome_file, 'wb') as genome_file:
        pickle.dump(best_genome, genome_file, pickle.HIGHEST_PROTOCOL)
    genome_file.close()
    os.replace(best_genome_file, "../SCS_Results/" + file + "/" + "best_genome.pickle")
        

    # Print experiment statistics
    print("\nBest ever fitness: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
    print("\nTrial elapsed time: %.3f sec" % (elapsed_time))
    print("Random seed:", seed)
    
        
    # Visualize best network's Genome
    winner_net_CPPN = NEAT.NeuralNetwork()
    gen_best_genome.BuildPhenotype(winner_net_CPPN)
    print("\nCPPN nodes: %d, connections: %d" % (len(winner_net_CPPN.neurons), len(winner_net_CPPN.connections)))

    # Visualize best network's phenotype
    winner_net = NEAT.NeuralNetwork()
    gen_best_genome.BuildESHyperNEATPhenotype(winner_net, exp.substrate, exp.params)
    winner_net.Save("../SCS_Results/" + file + "/winner_net.txt")
    


    with open("score.txt","w+") as score_file:
        score_file.write("\nBest ever fitness: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
        score_file.write("\nTrial elapsed time: %.3f sec" % (elapsed_time))
        score_file.write("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))
    score_file.close() 
    os.replace("score.txt", "../SCS_Results/" + file + "/" + "score.txt")

    #visualize.draw_net(winner_net, view=False, node_names=None, filename="substrate_graph", directory="../SCS_Results/" + file + "/", fmt='pdf')
    print("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))
    
    #save_params
    params_file = os.path.join(".", "params.pickle")
    with open(params_file, 'wb') as param_file1:
        pickle.dump(exp.params, param_file1)
    param_file1.close() 
    os.replace(params_file, "../SCS_Results/" + file + "/" +  "params.pickle")

    
    #save_substrate 
    substrate_file = os.path.join(".", "substrate.pickle")
    with open(substrate_file, 'wb') as substrate_file1:
        pickle.dump(exp.substrate, substrate_file1)
    substrate_file1.close() 
    os.replace(substrate_file, "../SCS_Results/" + file + "/" + "substrate.pickle")

    print('*******************************************')
    print("EVOLUTION DONE")
    print('*******************************************')
    
    
    import subprocess
    subprocess.Popen("python3 analysis.py", cwd="../SCS_Results/" + file + "/", shell=True) # used to be "Results/"
    os.rename('../SCS_Results/' + file,'../SCS_Results/' + file +'_'+ str(int(best_ever_goal_fitness))) # used to be "Results/" for both



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


    model = Multicellularity_model(net=net, depth=depth, height= g_exp.height, width= g_exp.width, 
    initial_cells=g_exp.initial_cells,
    energy = g_exp.energy,
    step_count = g_exp.step_count,
    nb_gap_junctions = g_exp.nb_gap_junctions,
    fitness= g_exp.fitness_function,
    goal = g_exp.goal,
    start_mat = g_exp.start_mat)
   

    model.verbose = False
    fitness_individual=0
    fitness_test = model.run_model(fitness_evaluation=True)
    if fitness_test>95:
        for i in range(10):
            fitness_individual += fitness_test
            fitness_test = model.run_model(fitness_evaluation=True)
            #net.Flush()
        fitness_individual = fitness_individual/10

        net.Save("../SCS_Results/" + file + "/winner_net_"+ str(fitness_individual)+".txt")
    else: 
        fitness_individual = fitness_test
        
    return fitness_individual












