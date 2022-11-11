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

def agents_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.state ==4:
        portrayal["Color"] = ["yellow"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.state ==3:
        portrayal["Color"] = ["grey"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.state ==1:
        portrayal["Color"] = ["blue"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.state ==2:
        portrayal["Color"] = ["red"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        

    return portrayal

def agents_portraya_state_tissue(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.state_tissue ==4:
        portrayal["Color"] = ["yellow"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.state_tissue ==3:
        portrayal["Color"] = ["grey"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.state_tissue ==1:
        portrayal["Color"] = ["blue"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.state_tissue ==2:
        portrayal["Color"] = ["red"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        

    return portrayal

def agents_portrayal2(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Cell and agent.stress ==0.0:
        portrayal["Color"] = ["#f9ebea"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1
        
    if type(agent) is Cell and agent.stress <=10  and agent.stress >0:
        portrayal["Color"] = ["#f2d7d5"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=20  and agent.stress >10:
        portrayal["Color"] = ["#e6b0aa"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=30 and agent.stress >20:
        portrayal["Color"] = ["#d98880"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    if type(agent) is Cell and agent.stress <=40  and agent.stress >30:
        portrayal["Color"] = ["#cd6155"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=50  and agent.stress >40:
        portrayal["Color"] = ["#c0392b"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1   
        
    if type(agent) is Cell and agent.stress <=60  and agent.stress >50:
        portrayal["Color"] = ["#a93226"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=70 and agent.stress >60:
        portrayal["Color"] = ["#922b21"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1  
    
    if type(agent) is Cell and agent.stress <=80  and agent.stress >70:
        portrayal["Color"] = ["#7b241c"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    

    if type(agent) is Cell and agent.stress <=90 and agent.stress >80:
        portrayal["Color"] = ["#641e16"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    if type(agent) is Cell and agent.stress >90:
        portrayal["Color"] = ["#1b2631"]
        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["r"] = 1    
        
    return portrayal

def agents_portrayal3(agent):
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

def agents_portrayal7(agent):
        if agent is None:
            return
    
        portrayal = {}
    
        if type(agent) is Cell :
            portrayal["Color"] = ["yellow"]
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = "true"
            portrayal["text"] = str(int(agent.molecules[0]))
            portrayal["text_color"] = ["black"]
            portrayal["Layer"] = 0
            portrayal["r"] = 0.1

        
        return portrayal
    

def get_elements(height, width):
    canvas_element = CanvasGrid(agents_portrayal, height, width, 200, 200)
    canvas_element2 = CanvasGrid(agents_portrayal2, height, width, 200, 200)
    canvas_element3 = CanvasGrid(agents_portrayal3, height, width, 200, 200)
    canvas_element7 = CanvasGrid(agents_portrayal7, height, width, 200, 200)
    canvas_element9 = CanvasGrid(agents_portraya_state_tissue, height, width, 200, 200)


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

    return [canvas_element, canvas_element9, canvas_element2, canvas_element3, canvas_element7, chart_element, chart_element1, chart_element2, chart_element3, chart_element4, chart_element5]

# If run as script.
if __name__ == '__main__':
    seed = int(np.load("seed.npy"))
    rng = NEAT.RNG()
    rng.Seed(seed)
    # Neural network of the winning genome
    net = NEAT.NeuralNetwork()
    net.Load("winner_net.txt")
    net.Flush()
    print("\nSubstrate nodes: %d, connections: %d" % (len(net.neurons), len(net.connections)))

    # Loading data
    sys.setrecursionlimit(100000)
    start = np.loadtxt("start_matrix.txt").astype(int).tolist()
    goal = np.loadtxt("goal_matrix.txt").astype(int).tolist()
    # Params for MESA model and the evolving neural network
    model_params = {}
    # exp = pickle.loads(exp)
    # how to load the object in???
    f = open("exp.pickle", 'rb')
    exp = pickle.load(f)
    f.close()
    model_params["ANN_inputs"] = exp.ANN_inputs
    model_params["ANN_outputs"] = exp.ANN_outputs
    #test
    print(model_params["ANN_inputs"])
    print(model_params["ANN_outputs"])
    print(exp.ANN_inputs)
    print(exp.ANN_outputs)
    print(exp.nb_ANN_outputs)
    print(exp.nb_ANN_inputs)
    # May remove general params soon...
    f = open("general_params.txt")
    for lines in f:
        items = lines.split(': ', 1)
        model_params[items[0]] = eval(items[1])
    f.close()
    model_params["net"] = net
    model_params["depth"] -= 1
    model_params["start"] = start
    model_params["goal"] = goal

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
