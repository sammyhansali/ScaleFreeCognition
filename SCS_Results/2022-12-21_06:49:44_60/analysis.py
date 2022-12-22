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

from psutil import process_iter
from signal import SIGKILL

def cell_types(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.cell_type[0] ==4:
        portrayal["Color"] = ["yellow"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.cell_type[0] ==3:
        portrayal["Color"] = ["black"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.cell_type[0] ==2:
        portrayal["Color"] = ["brown"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.cell_type[0] ==1:
        portrayal["Color"] = ["#C4A484"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        

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
    
        if type(agent) is Cell and agent.GJ_opening_ions ==0:
            portrayal["Color"] = ["grey"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1
            
        if type(agent) is Cell and agent.GJ_opening_ions <=0.25  and agent.GJ_opening_ions >0:
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

def GJ_opening_molecules(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        if type(agent) is Cell and agent.GJ_opening_molecs ==0:
            portrayal["Color"] = ["grey"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1
            
        if type(agent) is Cell and agent.GJ_opening_molecs <=0.25  and agent.GJ_opening_molecs >0:
            portrayal["Color"] = ["#173806"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.GJ_opening_molecs <=0.5  and agent.GJ_opening_molecs >0.25:
            portrayal["Color"] = ["#2d6e0c"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
    
        if type(agent) is Cell and agent.GJ_opening_molecs <=0.75  and agent.GJ_opening_molecs >0.5:
            portrayal["Color"] = ["#47b012"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
            
        if type(agent) is Cell  and agent.GJ_opening_molecs >0.75:
            portrayal["Color"] = ["#62f716"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["Layer"] = 0
            portrayal["r"] = 1    
        
        return portrayal

def molecules(agent):
        if agent is None:
            return
    
        portrayal = {}
        tot = 0
        n = len(agent.molecules)
        for i in range(n):
            tot+=agent.molecules[i][0]
    
        if type(agent) is Cell:
            portrayal["Color"] = ["yellow"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["text"] = str(int(tot))
            portrayal["text_color"] = ["black"]
            portrayal["Layer"] = 0
            portrayal["r"] = 0.1

        
        return portrayal

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
            portrayal["text"] = str(round(agent.energy[0],1))
            portrayal["text_color"] = ["black"]
            portrayal["Layer"] = 0
            portrayal["r"] = 0.1

        
        return portrayal
    

def get_elements(height, width):
    ce_cell_types = CanvasGrid(cell_types, height, width, 200, 200)
    # stress = CanvasGrid(stress, height, width, 200, 200) # Potential replaced stress's code
    ce_GJ_opening_ions = CanvasGrid(GJ_opening_ions, height, width, 200, 200)
    ce_molecules = CanvasGrid(molecules, height, width, 200, 200)
    ce_GJ_opening_molecules = CanvasGrid(GJ_opening_molecules, height, width, 200, 200)
    ce_potential = CanvasGrid(potential, height, width, 200, 200)
    ce_delta_potential = CanvasGrid(delta_potential, height, width, 200, 200)
    ce_energy = CanvasGrid(energy, height, width, 200, 200)
    # canvas_element9 = CanvasGrid(agents_portraya_state_tissue, height, width, 200, 200)


    chart_element = ChartModule(
        [{"Label": "Cells", "Color": "#AA0000"}], [{"Label": "Global stress", "Color": "##84e184"}]
    # [{"Label": "Global stress", "Color": "#AA0000"}], [{"Label": "Global", "Color": "##84e184"}]
    )

        
    chart_element1 = ChartModule(
            [{"Label": "Global stress", "Color": "blue"}]
    # [{"Label": "Global stress", "Color": "#AA0000"}], [{"Label": "Global", "Color": "##84e184"}]
    )

    chart_element2 = ChartModule(
            [{"Label": "Multicellularity", "Color": "green"}]
    # [{"Label": "Global stress", "Color": "#AA0000"}], [{"Label": "Global", "Color": "##84e184"}]
    )

    chart_element3 = ChartModule(
            [{"Label": "Entropy", "Color": "black"}]
    # [{"Label": "Global stress", "Color": "#AA0000"}], [{"Label": "Global", "Color": "##84e184"}]
    )

    chart_element4 = ChartModule(
            [{"Label": "Geometrical frustration", "Color": "black"}]
    # [{"Label": "Global stress", "Color": "#AA0000"}], [{"Label": "Global", "Color": "##84e184"}]
    )

    chart_element5 = ChartModule(
            [{"Label": "Internal stress", "Color": "black"}]
    # [{"Label": "Global stress", "Color": "#AA0000"}], [{"Label": "Global", "Color": "##84e184"}]
    )

    return [
                (ce_cell_types, ce_potential), # Cell States
                # ce_potential, # potential t1 (visualized w color)
                # delta_potential, # potential t1-t0
                ce_energy, # 
                ce_molecules,
                ce_GJ_opening_molecules,
                # canvas_element9, 
                # stress, # Agent's stress
                # ce_GJ_opening_ions, # GJ_opening_ions. Light = More open. Dark = More closed.
                # canvas_element7, # nb_molecs
                # chart_element, 
                # chart_element1, 
                # chart_element2, 
                # chart_element3, 
                # chart_element4, 
                # chart_element5,
            ]

# If run as script.
if __name__ == '__main__':
    seed = int(np.load("seed.npy"))
    rng = NEAT.RNG()
    rng.Seed(seed)
    # Neural network of the winning genome
    net = NEAT.NeuralNetwork()
    net.Load("winner_net.txt")
    net.Flush()
    # print("\nSubstrate nodes: %d, connections: %d" % (len(net.neurons), len(net.connections)))

    # Loading data
    sys.setrecursionlimit(100000)
    
    # Params for MESA model and the evolving neural network
    model_params = {}

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
    model_params["net"] = net
    model_params["depth"] -= 1
    model_params["start"] = start
    model_params["goal"] = goal
    model_params["bioelectric_stimulus"] = bioelectric_stimulus

    height=model_params["height"]
    width=model_params["width"]
    # Get chart and canvas elements
    elements = get_elements(height, width)

    # Test the winner network on the server
    import socketserver
    # with socketserver.TCPServer(("localhost", 55669), None) as s: #Getting "Adress already in use" error
    with socketserver.TCPServer(("localhost", 0), None) as s: #Getting "Adress already in use" error
        free_port = s.server_address[1] # Its grabbing the first free port it finds.
    server = ModularServer(
        Multicellularity_model, elements, "Multi-cellularity", model_params
    )
    server.port = free_port

    server.launch()
