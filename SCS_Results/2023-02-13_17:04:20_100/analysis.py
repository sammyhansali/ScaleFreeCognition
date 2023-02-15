from multicellularity.model_for_analysis import Multicellularity_model
try:
   import cPickle as pickle
except:
   import pickle
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from multicellularity.agents import Cell
import multicellularity.visualize as visualize
from run import eval_individual

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
import sys

from psutil import process_iter
from signal import SIGKILL

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

def test_eval_individual(genome, exp):
    
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
    
    net = NEAT.NeuralNetwork()
    genome.BuildESHyperNEATPhenotype(net, exp.substrate, exp.params)
    net.Flush()

    # model = Multicellularity_model(
    #     net = net, 
    #     exp = exp,
    # )
    # model.verbose = False
    # fit1 = model.run_model(fitness_evaluation=True)

    bb = 0
    for i in range(5):
        model = Multicellularity_model(net = net, exp = exp)
        # model = Multicellularity_model(
        #     net = net, 
        #     # exp = g_exp,
        #     depth = g_exp.MaxDepth,
        #     height = g_exp.height,
        #     width = g_exp.width,
        #     energy = g_exp.energy,
        #     step_count = g_exp.step_count,
        #     nb_output_molecules = g_exp.nb_output_molecules,
        #     goal = g_exp.goal,
        #     start = g_exp.start,
        #     bioelectric_stimulus = g_exp.bioelectric_stimulus,
        #     ANN_inputs = g_exp.ANN_inputs,
        #     ANN_outputs = g_exp.ANN_outputs,
        #     history_length = g_exp.history_length,
        #     e_penalty = g_exp.e_penalty,
        #     preset = g_exp.preset,
        #     multiple = g_exp.multiple,
        # )
        model.verbose = False
        bbb = model.run_model(fitness_evaluation=True)
        bb += bbb
        print(f"Run {i}: {bbb}")
    fit1=bb/5
    return fit1

### If run as script
if __name__ == '__main__':

    # ## Experiment file with all model parameters
    # with open("exp.pickle", "rb") as fp:
    #     exp = pickle.load(fp)

    ## Experimental - load genome instead of ANN
    # with open("best_genome.pickle", "rb") as bg:
    #     best_genome = pickle.load(bg)
    # print(f"Expected fitness (pickle):{test_eval_individual(best_genome, exp)}")

    # # Test
    # best_genome_2 = NEAT.Genome("best_genome.txt")
    # print(f"Expected fitness (txt):{test_eval_individual(best_genome_2, exp)}")

    # net = NEAT.NeuralNetwork()
    # best_genome.BuildESHyperNEATPhenotype(net, exp.substrate, exp.params)
    # net.Flush()

    ## Commented out below lines for test
    # seed = int(np.load("seed.npy"))
    # rng = NEAT.RNG()
    # rng.Seed(seed)
    
    # # Neural network of the winning genome
    # net = NEAT.NeuralNetwork()
    # net.Load("winner_net.txt")
    # net.Flush()

    # sys.setrecursionlimit(100000)

    ### Params for MESA model and the evolving neural network
    model_params = {}

    # model_params["net"] = net
    # exp.depth-=1      # Commenting out since should be 3 for run and 3 for analysis
    # model_params["exp"] = exp
    

    ### Tested to make sure exp is pickled accurately
    # model_params["depth"] = exp.params.MaxDepth
    # model_params['height'] = exp.height
    # model_params['width'] = exp.width
    # model_params['energy'] = exp.energy
    # model_params['step_count'] = exp.step_count
    # model_params['nb_output_molecules'] = exp.nb_output_molecules
    # model_params['goal'] = exp.goal
    # model_params['start'] = exp.start
    # model_params['bioelectric_stimulus'] = exp.bioelectric_stimulus
    # model_params['ANN_inputs'] = exp.ANN_inputs
    # model_params['ANN_outputs'] = exp.ANN_outputs
    # model_params['history_length'] = exp.history_length
    # model_params['e_penalty'] = exp.e_penalty
    # model_params['preset'] = exp.preset
    # model_params['multiple'] = exp.multiple

    # print(f"Depth {depth}, Height {height}, width {width}, energy {energy}, step_count {step_count}, nb_output_molecules {nb_output_molecules}")
    # print(f"goal {goal}, start {start}, bioelectric_stimulus {bioelectric_stimulus}, ANN_inputs {ANN_inputs}, ANN_outputs {ANN_outputs}, history_length {history_length}")
    # print(f"e_penalty {e_penalty}, preset {preset}, multiple {multiple}")
    ###

    # Matrices
    start = np.loadtxt("start_matrix.txt").astype(int).tolist()
    goal = np.loadtxt("goal_matrix.txt").astype(int).tolist()
    bioelectric_stimulus = np.loadtxt("bioelectric_stimulus_matrix.txt").tolist()
    if bioelectric_stimulus == 0:
        bioelectric_stimulus = None

    # ANN inputs and outputs
    with open("ANN_inputs.pickle", "rb") as f:
        model_params["ANN_inputs"] = pickle.load(f)
    with open("ANN_outputs.pickle", "rb") as f:
        model_params["ANN_outputs"] = pickle.load(f)
    f = open("general_params.txt")
    for lines in f:
        items = lines.split(': ', 1)
        model_params[items[0]] = eval(items[1])
    f.close()
    # model_params["net"] = net
    model_params["depth"] -= 1
    model_params["start"] = start
    model_params["goal"] = goal
    model_params["bioelectric_stimulus"] = bioelectric_stimulus

    height=model_params["height"]
    width=model_params["width"]
    nb_output_molecules = model_params["nb_output_molecules"]

    with open("best_genome.pickle", "rb") as bg:
        best_genome = pickle.load(bg)
    # print(f"Expected fitness (pickle):{test_eval_individual(best_genome, model_params)}")

    net = NEAT.NeuralNetwork()
    best_genome.BuildESHyperNEATPhenotype(net, exp.substrate, exp.params)
    net.Flush()
    model_params["net"] = net

    # ### Get chart and canvas elements
    # height = exp.height
    # width = exp.width
    # nb_output_molecules = exp.nb_output_molecules

    elements = get_elements(height, width, nb_output_molecules)

    ### Test the winner network on the server
    import socketserver
    with socketserver.TCPServer(("localhost", 0), None) as s: #Getting "Adress already in use" error
        free_port = s.server_address[1] # Its grabbing the first free port it finds.
    server = ModularServer(
        Multicellularity_model, elements, "Multi-cellularity", model_params
    )
    server.port = free_port

    server.launch()
