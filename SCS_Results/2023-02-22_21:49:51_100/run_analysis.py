import pickle
import MultiNEAT as NEAT
import os
# from analysis import sim

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

def test_eval_individual(net, exp):
    
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
    
    fit = 0
    for i in range(5):
        model = Multicellularity_model(net = net, exp = exp)
        model.verbose = False
        run = model.run_model(fitness_evaluation=True)
        fit += run
        print(f"Run {i}: {round(run,1)}")
    fit/=5
    return fit


def sim(exp, winner_net):
    ### Net approach (works by default, so stick with it)
    print(f"Expected fitness:{test_eval_individual(winner_net, exp)}")

    model_params = {}
    model_params["net"] = winner_net
    # exp.depth-=1
    model_params["exp"] = exp

    height = exp.height
    width = exp.width
    nb_output_molecules = exp.nb_output_molecules
    elements = get_elements(height, width, nb_output_molecules)

    ### Test the winner network on the server
    import socketserver
    with socketserver.TCPServer(("localhost", 0), None) as s: 
        free_port = s.server_address[1] 
    server = ModularServer(Multicellularity_model, elements, "Multi-cellularity", model_params)
    server.port = free_port
    server.launch()


# Getting absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Loading exp
with open(f"{current_dir}/exp.pickle", "rb") as fp:
    exp = pickle.load(fp)

# Loading Net
net = NEAT.NeuralNetwork()
net.Load(f"{current_dir}/winner_net.txt")
net.Flush()

# Simulation
sim(exp, net)