"""
Transition to multicellularity Model
================================

Léo Pio-Lopez
"""

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from multicellularity.schedule import RandomActivationByBreed
from multicellularity.agents import Cell
import seaborn as sns; sns.set_theme()
import numpy as np


limit = 1
multiplier = 20


class Multicellularity_model(Model):
    """
    Multicellularity Model
    """

    description = (
        "A model for simulating transition to multicellularity."
    )

    def __init__(self,
        # self,
        net=None,
        depth=None,
        height= None, # for the vizualization grid
        width= None,  # idem
        energy = None,
        step_count = None,
        # fitness =  None,
        nb_gap_junctions = None,
        goal = None,
        nb_goal_cells=None,
        start = None,
        fitness_evaluation=False,
        nb_output_molecules = None,
        ANN_inputs = [],
        ANN_outputs = [],
        full_reservoir = None, 
        depleted_reservoir = None,
        history_length = None,
        e_penalty = None,
        bioelectric_stimulus = None,
        start_dissimilarity = None,
        molecules_exchanged = None,
    ):
        """
        Create a new multicellularity model with the given parameters.

        Args:
            initial_cell: Number of cell to start with
            initial_toxines: Number of toxines to start with
            cell_gain_from_food: Energy a cell gains from eating a food
            grass: Whether to have the sheep eat grass for energy
            grass_regrowth_time: How long it takes for a grass patch to regrow
                                 once it is eaten
        """
        super().__init__()
        # Set parameters
        self.net=net
        self.depth=depth
        self.height = height
        self.width = width
        # self.initial_cells = initial_cells
        self.energy = energy
        self.step_count = step_count
        self.nb_gap_junctions = nb_gap_junctions
        # self.fitness = fitness
        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=False)
        self.goal = goal
        self.start = start
        self.history_length = history_length
        self.e_penalty = e_penalty
        self.bioelectric_stimulus = bioelectric_stimulus
        

        # Calculate number of cells in goal matrix
        mat = np.array(self.goal)
        self.nb_goal_cells = len(mat[mat.nonzero()])
        self.ANN_inputs = ANN_inputs
        self.ANN_outputs = ANN_outputs
        self.nb_output_molecules = nb_output_molecules

        # Data collection for charts in analysis
        datacollector_dict = {}
            # {
            #     # "Cells": lambda m: m.schedule.get_breed_count(Cell),
            #     # "Global stress": lambda m: m.schedule.get_global_stress(Cell),
            #     # "Multicellularity": lambda m: m.schedule.get_open_cells(Cell),
            #     #"Entropy": lambda m: m.schedule.get_spatial_entropy(Cell),
            #     # "Geometrical frustration": lambda m: m.schedule.general_geometrical_frustration(Cell),
            #     # "Internal stress": lambda m:m.schedule.get_internal_stress(Cell),
            #     # "Total molecules exchanged": lambda m:m.schedule.get_total_nb_molecules_exchanged(Cell),
            # }
        # for i in range(nb_output_molecules):
        #     datacollector_dict[f"Molecule {i} exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged(Cell, i)
        datacollector_dict["Molecule 0 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_0(Cell)
        datacollector_dict["Molecule 1 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_1(Cell)
        datacollector_dict["Molecule 2 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_2(Cell)
        # datacollector_dict["Molecule 3 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_3(Cell)
        datacollector_dict["Total molecules exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_tot(Cell)


        # if self.nb_output_molecules > 1:
        #     datacollector_dict["Total molecules exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged(Cell, "total")


        self.datacollector = DataCollector(datacollector_dict)

        # Create cells on the whole grid
        self.schedule.generate_cells(Cell)

        self.running = True
        if fitness_evaluation==False:
            self.datacollector.collect(self)

        # Starting dissimilarity
        self.start_dissimilarity = self.schedule.dissimilarity()


    # def get_nb_molecules_exchanged(self, Cell, i):
    #     # if i == 0:
    #     self.molecules_exchanged = self.schedule.nb_molecules_exchanged(Cell)  # Do calculations before getting 0th molec
    #     if i == "total":
    #         return sum(self.molecules_exchanged)
    #     # Otherwise, i is an integer that picks which molecule you want
    #     # print(i)
    #     return self.molecules_exchanged[i]
    
    # Don't delete, or you won't be able to "step" in the visualization in mesa....
    def step(self):
            self.schedule.step(Cell)
            self.datacollector.collect(self)
            # print(self.depleted_reservoir)


    # Model runner
    def run_model(self, fitness_evaluation):
        for i in range(self.step_count):
            self.step()

        if fitness_evaluation==True:
            self.fitness_score = self.schedule.global_fitness()  
            # self.fitness_score = self.schedule.global_fitness_3()  

        return self.fitness_score         
        
                

            
 
  
