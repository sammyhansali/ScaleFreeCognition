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

    # # Saving more nets
    # for i in range(2, 11):
    #     if len(best_genome) >= i:
    #         index = i * -1
    #         winner_net = NEAT.NeuralNetwork()
    #         # winner_net_2 = NEAT.NeuralNetwork()
    #         best_genome[index].BuildESHyperNEATPhenotype(winner_net, exp.substrate, exp.params)
    #         # best_genome[-2].BuildESHyperNEATPhenotype(winner_net_2, exp.substrate, exp.params)
    #         winner_net.Save(f"SCS_Results/{file}/winner_net_{i}.txt")
    #         # winner_net_2.Save("SCS_Results/" + file + "/winner_net_2.txt")
    
    # Pickle the experiment (needed for analysis)
    exp_file = os.path.join(".", "exp.pickle")
    with open(exp_file, 'wb') as f:
        pickle.dump(g_exp, f, pickle.HIGHEST_PROTOCOL)
    os.replace(exp_file, "SCS_Results/" + file + "/" +  "exp.pickle")
    ##### Below is experimental
    # Seeing if pickle is fucking up
    with open("SCS_Results/" + file + "/" +  "exp.pickle", "rb") as pp:
        test_pickled_exp = pickle.load(pp)

    
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

    ## Seeing if the unpickled exp fucks up the results
    g_exp = test_pickled_exp
    # g_exp = None # Line just to see if I'm actually changing the global variable
    # This does actually alter the global experiment file.
    ## Test lines above
    best_genome_2 = NEAT.Genome(fp)
    print("Expected 2:", best_fitness)
    fitty=0
    for i in range(5):
        th = eval_individual(best_genome_2)
        print(th)
        fitty+=th
    print("Actual", str(fitty/5))

    # best_genome_3 = NEAT.Genome(fp)
    # print("Expected 3:", best_fitness)
    # fitty=0
    # for i in range(5):
    #     th = eval_individual(best_genome_3)
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
    import subprocess

    # Activate the Conda environment using source activate
    # subprocess.Popen(['source', 'activate', 'mesamultineat'], shell=True) # Doesn't work, source not allowed i guess
    subprocess.Popen(['conda', 'activate', 'mesamultineat'], shell=False)
    # print("Python version (run):", sys.version_info)

    print("cc")
    # Run the analysis script using the Python executable from the Conda environment
    subprocess.Popen(['python', 'analysis.py'], cwd="SCS_Results/" + file + "/", shell=True) # used to be "Results/"
    print("bb")
    # subprocess.Popen('python analysis.py', cwd="SCS_Results/" + file + "/", shell=True) # used to be "Results/"
    os.rename('SCS_Results/' + file,'SCS_Results/' + file +'_'+ str(int(best_fitness))) # used to be "Results/" for both

    # ### Testing to see if works
    # model_params = {}
    # net = NEAT.NeuralNetwork()
    # best_genome.BuildESHyperNEATPhenotype(net, exp.substrate, exp.params)
    # net.Flush()
    # model_params["net"] = net
    # model_params["exp"] = exp

    # ### Get chart and canvas elements
    # height = exp.height
    # width = exp.width
    # nb_output_molecules = exp.nb_output_molecules

    # elements = get_elements(height, width, nb_output_molecules)

    # ### Test the winner network on the server
    # import socketserver
    # with socketserver.TCPServer(("localhost", 0), None) as s: #Getting "Adress already in use" error
    #     free_port = s.server_address[1] # Its grabbing the first free port it finds.
    # server = ModularServer(
    #     Multicellularity_model, elements, "Multi-cellularity", model_params
    # )
    # server.port = free_port

    # server.launch()


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

def cell_types(agent):
    if agent is None:
        return

    portrayal = {}
    if type(agent) is Cell:
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        # portrayal["w"] = 1
        # portrayal["h"] = 1

    if agent.cell_type[0] ==4:
        portrayal["Color"] = ["black"]
        
        
    if agent.cell_type[0] ==3:
        portrayal["Color"] = ["blue"]
        
    if agent.cell_type[0] ==2:
        portrayal["Color"] = ["brown"]

    if agent.cell_type[0] ==1:
        portrayal["Color"] = ["#C4A484"]
        

    return portrayal


def stress(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.stress[0] ==0.0:
        portrayal["Color"] = ["#f9ebea"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.stress[0] <=10  and agent.stress[0] >0:
        portrayal["Color"] = ["#f2d7d5"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress[0] <=20  and agent.stress[0] >10:
        portrayal["Color"] = ["#e6b0aa"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress[0] <=30 and agent.stress[0] >20:
        portrayal["Color"] = ["#d98880"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    if type(agent) is Cell and agent.stress[0] <=40  and agent.stress[0] >30:
        portrayal["Color"] = ["#cd6155"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress[0] <=50  and agent.stress[0] >40:
        portrayal["Color"] = ["#c0392b"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1   
        
    if type(agent) is Cell and agent.stress[0] <=60  and agent.stress[0] >50:
        portrayal["Color"] = ["#a93226"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress[0] <=70 and agent.stress[0] >60:
        portrayal["Color"] = ["#922b21"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1  
    
    if type(agent) is Cell and agent.stress[0] <=80  and agent.stress[0] >70:
        portrayal["Color"] = ["#7b241c"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress[0] <=90 and agent.stress[0] >80:
        portrayal["Color"] = ["#641e16"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    if type(agent) is Cell and agent.stress[0] >90:
        portrayal["Color"] = ["#1b2631"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    return portrayal

def GJ_opening_ions(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        # if type(agent) is Cell and agent.GJ_opening_ions ==0:
        #     portrayal["Color"] = ["grey"]
        #     portrayal["Shape"] = "circle"
        #     portrayal["Filled"] = "true"
        #     portrayal["Layer"] = 0
        #     portrayal["r"] = 1
            
        if type(agent) is Cell and agent.GJ_opening_ions <=0.25  and agent.GJ_opening_ions >=0:
            portrayal["Color"] = ["#173806"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.GJ_opening_ions <=0.5  and agent.GJ_opening_ions >0.25:
            portrayal["Color"] = ["#2d6e0c"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.GJ_opening_ions <=0.75  and agent.GJ_opening_ions >0.5:
            portrayal["Color"] = ["#47b012"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
            
        if type(agent) is Cell  and agent.GJ_opening_ions >0.75:
            portrayal["Color"] = ["#62f716"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
        
        return portrayal

def GJ_generator(GJ):
    portrayal = {}
    
    portrayal["Shape"] = "circle"
    portrayal["Filled"] = "true"
    portrayal["Layer"] = 0
    portrayal["r"] = 1
    portrayal["text"] = str(round(GJ, 1))
    portrayal["text_color"] = ["black"]


    if GJ == 0:
        portrayal["Color"] = ["grey"]
    elif GJ <=0.25  and GJ >0:
        portrayal["Color"] = ["#173806"]
    elif GJ <=0.5  and GJ >0.25:
        portrayal["Color"] = ["#2d6e0c"]
    elif GJ <=0.75  and GJ >0.5:
        portrayal["Color"] = ["#47b012"]
    else:
        portrayal["Color"] = ["#62f716"]
    
    return portrayal

def GJ_molecules_0(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    GJ = agent.GJ_molecules[0]
    return GJ_generator(GJ)


def GJ_molecules_1(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    GJ = agent.GJ_molecules[1]
    return GJ_generator(GJ)

def GJ_molecules_2(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    GJ = agent.GJ_molecules[2]
    return GJ_generator(GJ)

def GJ_molecules_3(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    GJ = agent.GJ_molecules[3]
    return GJ_generator(GJ)

def GJ_molecules_4(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    GJ = agent.GJ_molecules[4]
    return GJ_generator(GJ)

def molecules_generator(molecules):
    portrayal = {}

    portrayal["Shape"] = "circle"
    portrayal["Filled"] = "true"
    portrayal["text"] = str(int(molecules))
    portrayal["text_color"] = ["black"]
    portrayal["Layer"] = 0
    portrayal["r"] = 1

    if molecules >=  10 :
        portrayal["Color"] = ["#014fff"]
    elif molecules >= 5:
        portrayal["Color"] = ["#00b5ff"]
    else:
        portrayal["Color"] = ["#bff8fd"]
    
    return portrayal
    
def molecules(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    molecules = 0
    n = len(agent.molecules)
    for i in range(n):
        molecules+=agent.molecules[i][0]
    return molecules_generator(molecules)


def molecules_0(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    molecules = agent.molecules[0][0]
    return molecules_generator(molecules)


def molecules_1(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    molecules = agent.molecules[1][0]
    return molecules_generator(molecules)

def molecules_2(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    molecules = agent.molecules[2][0]
    return molecules_generator(molecules)

def molecules_3(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    molecules = agent.molecules[3][0]
    return molecules_generator(molecules)

def molecules_4(agent):
    if agent is None:
        return
    if type(agent) is not Cell:
        return {}

    molecules = agent.molecules[4][0]
    return molecules_generator(molecules)


def potential(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.potential[0] == 0.0:
        portrayal["Color"] = ["#ff7800"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.potential[0] >= -10  and agent.potential[0] < 0:
        portrayal["Color"] = ["#ea6700"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.potential[0] >= -20  and agent.potential[0] < -10:
        portrayal["Color"] = ["#d45500"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.potential[0] >= -30 and agent.potential[0] < -20:
        portrayal["Color"] = ["#bf4400"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    if type(agent) is Cell and agent.potential[0] >= -40  and agent.potential[0] < -30:
        portrayal["Color"] = ["#ab3200"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.potential[0] >= -50  and agent.potential[0] < -40:
        portrayal["Color"] = ["#971f00"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1   
        
    if type(agent) is Cell and agent.potential[0] >= -60  and agent.potential[0] < -50:
        portrayal["Color"] = ["#840500"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.potential[0] >= -70 and agent.potential[0] < -60:
        portrayal["Color"] = ["#720000"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1  
    
    if type(agent) is Cell and agent.potential[0] >= -80  and agent.potential[0] < -70:
        portrayal["Color"] = ["#610000"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.potential[0] >= -90 and agent.potential[0] < -80:
        portrayal["Color"] = ["#520000"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
        
    if type(agent) is Cell and agent.potential[0] < -90:
        portrayal["Color"] = ["#000000"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    return portrayal

# potential
# def potential(agent):
#         if agent is None:
#             return
    
#         portrayal = {}
    
#         if type(agent) is Cell:
#             portrayal["Color"] = ["yellow"]
#             portrayal["Shape"] = "circle"
#             portrayal["Filled"] = "true"
#             portrayal["text"] = str(round(agent.potential[0],1))
#             portrayal["text_color"] = ["black"]
#             portrayal["Layer"] = 0
#             portrayal["r"] = 0.1

        
#         return portrayal

# potential T1
def delta_potential(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        if type(agent) is Cell :
            portrayal["Color"] = ["yellow"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["text"] = str(round(agent.potential[0]-agent.potential[1], 1))
            portrayal["text_color"] = ["black"]
            portrayal["Layer"] = 0
            portrayal["r"] = 0.1

        
        return portrayal

def energy(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        if type(agent) is Cell :
            portrayal["Color"] = ["yellow"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["text"] = str(round(agent.energy[0]))
            portrayal["text_color"] = ["black"]
            portrayal["Layer"] = 0
            portrayal["r"] = 0.1

        
        return portrayal
    
def energy_delta(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        if type(agent) is Cell :
            portrayal["Color"] = ["yellow"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["text"] = str(round(agent.energy[0]-agent.energy[1],1))
            portrayal["text_color"] = ["black"]
            portrayal["Layer"] = 0
            portrayal["r"] = 0.1

        
        return portrayal

def get_elements(height, width, nb_output_molecules):
    return_list =   [
                        CanvasGrid(cell_types, height, width, 200, 200),
                        CanvasGrid(potential, height, width, 200, 200),
                    ]

    molec_list = [molecules_0, molecules_1, molecules_2, molecules_3, molecules_4]
    GJ_molec_list = [GJ_molecules_0, GJ_molecules_1, GJ_molecules_2, GJ_molecules_3, GJ_molecules_4]
    for i in range(nb_output_molecules):
        return_list.append(
                CanvasGrid(molec_list[i], height, width, 200, 200)
            )
        return_list.append(
                CanvasGrid(GJ_molec_list[i], height, width, 200, 200)
            )
    others = [molecules, energy, energy_delta]
    for i in others:
        return_list.append(CanvasGrid(i, height, width, 200, 200))

    # Making charts for molecule exchange activity
    for i in range(nb_output_molecules):
        return_list.append(
            ChartModule(
                [{"Label": f"Molecule {i} exchanged", "Color": "#AA0000"}]
            )
        )
    if nb_output_molecules > 1:
        return_list.append(
            ChartModule(
                [{"Label": f"Total molecules exchanged", "Color": "#AA0000"}]
            )
        )
    # Chart for fitness
    return_list.append(
        ChartModule(
            [{"Label": "Fitness", "Color": "#AA0000"}]
        )
    )

    return return_list