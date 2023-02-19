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
    # best_genome = None
    best_genome = []
    best_fitness = -20000
    best_ID = -1

    for generation in range(exp.nb_gens):
        gen_time = time.time()

        # Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)
        # fitnesses = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)
        # [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        ## Commenting 6 lines for brief test
        ff = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)
        ## Commenting 4 lines for brief test
        # ff = np.array(ff)
        # for z in range(9):
        #     ff += np.array(EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number))
        # ff /= 10
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, ff)]

        # ### Trying this out instead of inbuilt
        # fitnesses = [fit for index,fit in yolo(genome_list, eval_individual, display=False, cores=cpu_number)]
        # fitnesses = np.array(fitnesses)
        # for z in range(9):
        #     fitnesses += np.array(EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number))
        # fitnesses /= 10

        ### Printing
        # if generation == exp.nb_gens-1:
        #     print(ff)

        ## Commenting out below since using default function now.
        # if generation == exp.nb_gens-1:
        #     # print(ff)
        #     print(fitnesses)
        #     # print(ff == [fit for index,fit in fitnesses])
        # [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        gen_best_genome = pop.Species[0].GetLeader()   
        gen_best_fitness = gen_best_genome.GetFitness()
        gen_best_ID = gen_best_genome.GetID()

        

        if (gen_best_fitness > best_fitness):
            # best_genome = gen_best_genome
            best_genome.append(gen_best_genome)
            best_fitness = gen_best_fitness
            best_ID = gen_best_ID

        # Take winning genome, test again to make sure it is valid.
        # if generation == exp.nb_gens-1:
            # print("Expected:", best_fitness)
            # fitty=0
            # for i in range(10):
            #     # th = eval_individual(best_genome)
            #     th = eval_individual(best_genome[-1])
            #     print(th)
            #     fitty+=th
            # print("Actual", str(fitty/20))
        
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
    print("Expected:", best_fitness)
    fitty=0
    for i in range(1):
        th = eval_individual(bst)
        print(th)
        fitty+=th
    print("Actual", str(fitty/1))

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
    # os.replace(best_genome_file, "SCS_Results/" + file + "/" + "best_genome.pickle")

    # Visualize best network's Genome
    winner_net_CPPN = NEAT.NeuralNetwork()
    bst.BuildPhenotype(winner_net_CPPN)
    # print("\nCPPN nodes: %d, connections: %d" % (len(winner_net_CPPN.neurons), len(winner_net_CPPN.connections)))

    # Visualize best network's Phenotype
    winner_net = NEAT.NeuralNetwork()
    bst.BuildESHyperNEATPhenotype(winner_net, exp.substrate, exp.params)
    winner_net.Save("SCS_Results/" + file + "/winner_net.txt")
    # print("\nSubstrate nodes: %d, connections: %d" % (len(winner_net.neurons), len(winner_net.connections)))

    
    # Pickle the experiment (needed for analysis)
    exp_file = os.path.join(".", "exp.pickle")
    with open(exp_file, 'wb') as f:
        pickle.dump(exp, f, pickle.HIGHEST_PROTOCOL)
    os.replace(exp_file, "SCS_Results/" + file + "/" +  "exp.pickle")
    ##### Below is experimental
    # Seeing if pickle is fucking up
    # with open("SCS_Results/" + file + "/" +  "exp.pickle", "rb") as pp:
    #     test_pickled_exp = pickle.load(pp)

    # print("Pickled exp equals original exp: ", test_pickled_exp==exp)
    # print("Pickled substrate equals original substrate: ", test_pickled_exp.substrate==exp.substrate)
    # print("Pickled params equals original params: ", test_pickled_exp.params==exp.params)
    
    # # Pickling substrate
    # substrate_file = os.path.join(".", "substrate.pickle")
    # with open(substrate_file, 'wb') as substrate_file1:
    #     pickle.dump(exp.substrate, substrate_file1)
    # substrate_file1.close() 
    # os.replace(substrate_file, "SCS_Results/" + file + "/" + "substrate.pickle")
    # # Seeing if pickle is fucking up
    # with open("SCS_Results/" + file + "/" +  "substrate.pickle", "rb") as pp:
    #     test_pickled_substrate = pickle.load(pp)
    # print("Pickled substrate equals original substrate: ", test_pickled_substrate==exp.substrate)

    ### Trying to save only certain exp attributes
    # selected_attrs = {
    #     "depth": exp.params.MaxDepth,
    #     "height": exp.height,
    #     "width": exp.width,
    #     "energy": exp.energy,
    #     "step_count": exp.step_count,
    #     "nb_output_molecules": exp.nb_output_molecules,
    #     "goal": exp.goal,
    #     "start": exp.start,
    #     "bioelectric_stimulus": exp.bioelectric_stimulus,
    #     "ANN_inputs": exp.ANN_inputs,
    #     "ANN_outputs": exp.ANN_outputs,
    #     "history_length": exp.history_length,
    #     "e_penalty": exp.e_penalty,
    #     "preset": exp.preset,
    #     "multiple": exp.multiple,
    # }
    # with open("SCS_Results/" + file + "/" +  "exp.txt", "w") as f:
    #     pickle.dump(selected_attrs, f)
    ##### End of experimental

    ## Here the experiment is normal, not the unpickled one
    fp = "SCS_Results/" + file + "/" + "best_genome.txt"
    best_genome = NEAT.Genome(fp)
    print("Expected 1:", best_fitness)
    fitty=0
    for i in range(5):
        th = eval_individual(best_genome)
        print(th)
        fitty+=th
    print("Actual", str(fitty/5))

    # ## Seeing if the unpickled exp fucks up the results
    # g_exp = test_pickled_exp
    # # g_exp = None # Line just to see if I'm actually changing the global variable
    # # This does actually alter the global experiment file.
    # ## Test lines above
    # best_genome_2 = NEAT.Genome(fp)
    # print("Expected 2:", best_fitness)
    # fitty=0
    # for i in range(5):
    #     th = eval_individual(best_genome_2)
    #     print(th)
    #     fitty+=th
    # print("Actual", str(fitty/5))

#     ### Test, saving data that I used to save
#     Result_file = "SCS_Results/" + file + "/"+ "general_params.txt"
#     with open("general_params.txt","w+") as general_params_file:
#         print("depth: %s  \n\
# height: %s  \n\
# width: %s  \n\
# energy: %s \n\
# nb_output_molecules: %s  \n\
# history_length: %s  \n\
# e_penalty: %s  \n\
# preset: %s  \n\
# multiple: %s  \n\
# step_count: %s" % (exp.depth, exp.height, exp.width, 
#         exp.energy, exp.nb_output_molecules, exp.history_length, exp.e_penalty, exp.preset, exp.multiple, exp.step_count), 
#         file=general_params_file)
#     os.replace("general_params.txt", Result_file)
#     # Matrices
#     np.savetxt("SCS_Results/" + file + "/start_matrix.txt", exp.start)
#     np.savetxt("SCS_Results/" + file + "/goal_matrix.txt", exp.goal)
#     if exp.bioelectric_stimulus is not None:
#         np.savetxt("SCS_Results/" + file + "/bioelectric_stimulus_matrix.txt", exp.bioelectric_stimulus)
#     else:
#         np.savetxt("SCS_Results/" + file + "/bioelectric_stimulus_matrix.txt", np.zeros(1)) # Blank
#     #save_params
#     params_file = os.path.join(".", "params.pickle")
#     with open(params_file, 'wb') as param_file1:
#         pickle.dump(exp.params, param_file1)
#     param_file1.close() 
#     os.replace(params_file, "SCS_Results/" + file + "/" +  "params.pickle")
#     # ANN inputs
#     inp_file = os.path.join(".", "ANN_inputs.pickle")
#     with open(inp_file, 'wb') as f:
#         pickle.dump(exp.ANN_inputs, f)
#     os.replace(inp_file, "SCS_Results/" + file + "/" +  "ANN_inputs.pickle")
#     # ANN outputs
#     out_file = os.path.join(".", "ANN_outputs.pickle")
#     with open(out_file, 'wb') as f:
#         pickle.dump(exp.ANN_outputs, f)
#     os.replace(out_file, "SCS_Results/" + file + "/" +  "ANN_outputs.pickle")
    
#     #save_substrate 
#     substrate_file = os.path.join(".", "substrate.pickle")
#     with open(substrate_file, 'wb') as substrate_file1:
#         pickle.dump(exp.substrate, substrate_file1)
#     substrate_file1.close() 
#     os.replace(substrate_file, "SCS_Results/" + file + "/" + "substrate.pickle")
#     ###



    # Run analysis for visualization
    # import subprocess

    # # Activate the Conda environment using source activate
    # # subprocess.Popen(['source', 'activate', 'mesamultineat'], shell=True) # Doesn't work, source not allowed i guess
    # subprocess.Popen(['conda', 'activate', 'mesamultineat'], shell=False) # Keep conda, keep false
    # # print("Python version (run):", sys.version_info)

    # print("cc")
    # # Run the analysis script using the Python executable from the Conda environment
    # subprocess.Popen(['python', 'analysis.py'], cwd="SCS_Results/" + file + "/", shell=True) # used to be "Results/"
    # print("bb")
    # # subprocess.Popen('python analysis.py', cwd="SCS_Results/" + file + "/", shell=True) # used to be "Results/"
    os.rename('SCS_Results/' + file,'SCS_Results/' + file +'_'+ str(int(best_fitness))) # used to be "Results/" for both

    # # print("Pickle version (run):", pickle.format_version)
    # import subprocess

    # # Activate the conda environment and run the command
    # # subprocess.Popen(['conda', 'run', '-n', 'mesamultineat', 'python', 'analysis.py'], cwd="SCS_Results/" + file, shell=True)

    # # Activate the conda environment using the shell script
    # # subprocess.Popen(['bash', '-c', 'source activate_conda.sh'])

    # # Run the command using the activated environment
    # subprocess.run(['python', 'analysis.py'], cwd="SCS_Results/" + file +'_'+ str(int(best_fitness)))
    print(best_genome.GetFitness())
    print(bst.GetFitness())
    from analysis import sim
    sim(exp, best_genome)

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
