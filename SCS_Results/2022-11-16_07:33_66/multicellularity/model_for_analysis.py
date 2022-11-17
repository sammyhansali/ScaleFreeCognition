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
                "Geometrical frustration": lambda m: m.schedule.geometrical_frustration(Cell),
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




              

    # French flag step
    def step(self, fitness_evaluation=True):
        
        reward_mat, stress_mat=self.schedule.reward_by_patches()
        # perc_blue, perc_red, perc_white = self.schedule.percentages_french_flag()    
        # fitness_ff=self.schedule.french_flag()
        fitness_score=self.schedule.fitness()
        tissue_matrix, state_matrix, stress_matrix, energy_matrix, molecules_matrix = self.schedule.adaptive_tissue()

        # self.schedule.step(Cell, reward_mat, stress_mat, perc_blue, perc_red, perc_white, fitness_score, tissue_matrix )
        self.schedule.step(Cell, reward_mat, stress_mat, fitness_score, tissue_matrix )
        self.schedule.update_state_tissue_costStateChange()
        
        # if fitness_evaluation== False:
        #     mol=0
        #     for i in range(self.height):
        #        for j in range(self.width):
                   
        #            if len(self.grid.get_cell_list_contents([(j,i)]))>0:
        #                cell = self.grid.get_cell_list_contents([(j,i)])[0]  
        #                mol+=cell.molecules[0]        
            
        # collect data
        self.datacollector.collect(self)
        
        # if fitness_evaluation==True:
        #     self.traj.append(self.schedule.get_pca_goal(Cell))
        #     self.traj1.append(self.schedule.scientific_pca(Cell))
        
    # Simple Cross step
    
    # Model runner
    def run_model(self, fitness_evaluation):
        for i in range(self.step_count):
            self.step(fitness_evaluation)

        if fitness_evaluation==True:
    
            # self.fitness_score=0
            # # general_energy=0
            # # Not using schedule.fitness() because want to measure g_e as well. 
            # for i in range(self.height):
            #      for j in range(int(self.width)):
            #          if len(self.grid.get_cell_list_contents([(j,i)]))>0:
            #              cell = self.grid.get_cell_list_contents([(j,i)])[0]
            #              if cell.state_tissue == cell.goal:
            #                  self.fitness_score+=1
            #             #  general_energy+=cell.energy
            # # print(self.fitness_score)
                             
            # self.fitness_score = self.fitness_score/(self.nb_goal_cells)*100
            # # general_energy   = general_energy/ (self.nb_goal_cells*10)
            self.fitness_score = self.schedule.fitness()  

            # remaining_cells = self.schedule.get_breed_count(Cell)
            # if remaining_cells == 0:
            #     remaining_cells=1
    
        return self.fitness_score         
        
                

            
 
  
