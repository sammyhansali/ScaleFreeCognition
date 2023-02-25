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
        traj = [],
        traj1=[],
        states_matrix = [],
        stress_matrix=[],
        energy_matrix  = [],
        molecules_matrix  = [],
        fitness_evaluation=False,
        nb_output_molecules = None,
        ANN_inputs = [],
        ANN_outputs = [],
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
        self.traj = traj
        self.traj1 = traj1
        self.states_matrix = states_matrix
        self.stress_matrix = stress_matrix
        self.energy_matrix = energy_matrix
        self.molecules_matrix = molecules_matrix
        # self.nb_output_molecules = nb_output_molecules
        self.datacollector = DataCollector(
            {

                # "Cells": lambda m: m.schedule.get_breed_count(Cell),
                "Global stress": lambda m: m.schedule.get_global_stress(Cell),
                "Multicellularity": lambda m: m.schedule.get_open_cells(Cell),
                #"Entropy": lambda m: m.schedule.get_spatial_entropy(Cell),
                "Geometrical frustration": lambda m: m.schedule.general_geometrical_frustration(Cell),
                "Internal stress": lambda m:m.schedule.get_internal_stress(Cell),


            }
        )
        # Calculate number of cells in goal matrix
        mat = np.array(self.goal)
        self.nb_goal_cells = len(mat[mat.nonzero()])
        self.ANN_inputs = ANN_inputs
        self.ANN_outputs = ANN_outputs

        # Create cells on the whole grid
        self.schedule.generate_cells(Cell, self.start)


        self.running = True
        if fitness_evaluation==False:
            self.datacollector.collect(self)

    
    # Don't delete, or for some reason the mesa visualization will not work properly...
    def step(self, depleted_reservoir, full_reservoir):
            # fitness_score = self.schedule.fitness()
            depleted_reservoir = self.schedule.step(Cell, depleted_reservoir, full_reservoir)
            self.datacollector.collect(self)
            print(depleted_reservoir)
            return depleted_reservoir


    # Model runner
    def run_model(self, fitness_evaluation):
        # If you want to make infinite and finite energy reservoirs, they have to be right here.
        # Infinite E reservoir is implicit, so I will only define finite here.
        full_reservoir = 100 
        depleted_reservoir = full_reservoir
        #Dummy value for now. Represents available energy of tissue to get the morphogenesis done.

        # Steps
        for i in range(self.step_count):
            depleted_reservoir = self.step(depleted_reservoir, full_reservoir)
            # print(depleted_reservoir)

        if fitness_evaluation==True:
            self.fitness_score = self.schedule.fitness()  

        return self.fitness_score         
        
                

            
 
  
