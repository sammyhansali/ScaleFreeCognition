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
from multicellularity.update_model import UpdateModel
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
        net=None,
        depth=None,
        height= None, # for the vizualization grid
        width= None,  # idem
        energy = None,
        step_count = None,
        global_fitness = None,
        goal = None,
        nb_goal_cells=None,
        start = None,
        fitness_evaluation=False,
        nb_output_molecules = None,
        ANN_inputs = [],
        ANN_outputs = [],
        history_length = None,
        e_penalty = None,
        bioelectric_stimulus = None,
        start_dissimilarity = None,
        molecules_exchanged = None,
        preset = None,
        multiple = None,
        exp = None,
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
        self.net = net
        self.exp = exp

        ## Setting parameters from experiment config file
        # self.depth = exp.depth
        self.depth = exp.params.MaxDepth
        self.height = exp.height
        self.width = exp.width
        self.energy = exp.energy
        self.step_count = exp.step_count
        self.nb_output_molecules = exp.nb_output_molecules
        self.goal = exp.goal
        self.start = exp.start
        self.bioelectric_stimulus = exp.bioelectric_stimulus
        self.ANN_inputs = exp.ANN_inputs
        self.ANN_outputs = exp.ANN_outputs
        self.history_length = exp.history_length
        self.e_penalty = exp.e_penalty
        self.preset = exp.preset
        self.multiple = exp.multiple

        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=False)
        self.global_fitness = [0]*self.history_length
        

        # Calculate number of cells in goal matrix
        mat = np.array(self.goal)
        self.nb_goal_cells = len(mat[mat.nonzero()])
        # self.ANN_inputs = ANN_inputs
        # self.ANN_outputs = ANN_outputs

        # Data collection for charts in analysis
        datacollector_dict = {
            "Fitness": lambda m:m.get_fitness()
        }
        for i in range(self.nb_output_molecules):
            if i==0:
                datacollector_dict["Molecule 0 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_0(Cell)
            elif i==1:
                datacollector_dict["Molecule 1 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_1(Cell)
            elif i==2:
                datacollector_dict["Molecule 2 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_2(Cell)
            elif i==3:
                datacollector_dict["Molecule 3 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_3(Cell)
            elif i==4:
                datacollector_dict["Molecule 4 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_4(Cell)
        if self.nb_output_molecules > 1:
            datacollector_dict["Total molecules exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_tot(Cell)
        self.datacollector = DataCollector(datacollector_dict)

        # Create cells on the whole grid
        self.schedule.generate_cells(Cell, self.exp.start_molecs)
        self.start_dissimilarity = self.schedule.dissimilarity()
        self.update_global_fitness()

        self.running = True
        if fitness_evaluation==False:
            self.datacollector.collect(self)

        # Updating model
        self.update_model = UpdateModel()   # Class

    # For datacollector
    def get_fitness(self):
        return self.global_fitness[0]

    def update_history(self, var, update):
        return [update] + var[:-1]

    def update_global_fitness(self):
        new_fit = self.schedule.global_fitness()/100
        # new_fit = self.schedule.organ_focused_fitness()/100
        self.global_fitness = self.update_history(self.global_fitness, new_fit)

    # Don't delete, or you won't be able to "step" in the visualization in mesa....
    def step(self):
        self.update_global_fitness()
        if self.preset is not None:
            for p in self.preset:
                getattr(self.update_model, p[0])(self, p[1])
        # getattr(self.update_model, self.preset)(self)
        self.schedule.step(Cell)
        self.datacollector.collect(self)

    # Model runner
    def run_model(self):
        for i in range(self.step_count):
            self.step()

        self.fitness_score = self.schedule.global_fitness()
        # return self.schedule.global_fitness()  
        # return self.schedule.organ_focused_fitness()
        return self.fitness_score

                

            
 
  
