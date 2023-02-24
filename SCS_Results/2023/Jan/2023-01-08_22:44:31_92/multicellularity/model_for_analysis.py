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
        substrate = None,
        params = None,
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
        self.depth = 3
        self.height = 9
        self.width = 9
        # self.initial_cells = initial_cells
        self.energy = 50
        self.step_count = 100
        self.nb_gap_junctions = nb_gap_junctions
        # self.fitness = fitness
        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=False)
        goal =     [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [ 1, 3, 3, 1, 1, 1, 3, 3, 1],
                    [ 1, 3, 1, 1, 1, 1, 1, 3, 1],
                    [ 1, 1, 1, 2, 1, 2, 1, 1, 1],
                    [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [ 1, 1, 2, 2, 2, 2, 2, 1, 1],
                    [ 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        self.goal = goal[::-1]
        
        start =    [[ 3, 3, 1, 1, 1, 1, 1, 1, 1],
                    [ 3, 1, 1, 1, 1, 1, 1, 1, 1],
                    [ 1, 1, 2, 1, 1, 1, 1, 1, 1],
                    [ 1, 1, 1, 1, 2, 1, 1, 1, 1],
                    [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [ 1, 2, 2, 2, 2, 2, 1, 1, 1],
                    [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [ 1, 1, 1, 1, 1, 1, 1, 3, 3],
                    [ 1, 1, 1, 1, 1, 1, 1, 3, 1]]
        self.start = start[::-1]

        self.nb_output_molecules = 2

        self.history_length = 2
        self.e_penalty = 0.95

        electric = [[ -80, -80, -80, -80, -80, -80, -80, -80, -80],
                    [ -80, -80, -80, -80, -80, -80, -80, -80, -80],
                    [ -80, -10, -10, -80, -80, -80, -10, -10, -80],
                    [ -80, -10, -80, -80, -80, -80, -80, -10, -80],
                    [ -80, -80, -80, -50, -80, -50, -80, -80, -80],
                    [ -80, -80, -80, -80, -80, -80, -80, -80, -80],
                    [ -80, -80, -80, -80, -80, -80, -80, -80, -80],
                    [ -80, -80, -50, -50, -50, -50, -50, -80, -80],
                    [ -80, -80, -80, -80, -80, -80, -80, -80, -80]]
        self.bioelectric_stimulus = electric[::-1]

        # Calculate number of cells in goal matrix
        mat = np.array(self.goal)
        self.nb_goal_cells = len(mat[mat.nonzero()])
        ANN_inputs=     [       
                            "bias",
                            "pos_x",
                            "pos_y",
                            # "goal_cell_type",
                        ]
        ANN_inputs.extend(["potential"]*self.history_length) 
        ANN_inputs.extend(["cell_type"]*self.history_length)
        ANN_inputs.extend(["energy"]*self.history_length)
        ANN_inputs.extend(["molecules"]*self.history_length*self.nb_output_molecules) 
        ANN_inputs.extend(["global_fitness"]*self.history_length)
        self.ANN_inputs = ANN_inputs

        ANN_outputs=    [    
                            # "GJ_opening_ions", 
                            # "charge_to_send",
                            # "cell_type",
                            # "GJ_opening_molecs",
                        ] 
        for i in range(self.nb_output_molecules):
                ANN_outputs.append(f"molecule_{i}_to_send")
                ANN_outputs.append(f"molecule_{i}_GJ")
        self.ANN_outputs = ANN_outputs

        self.nb_output_molecules = 2

        # Data collection for charts in analysis
        datacollector_dict = {}
        for i in range(self.nb_output_molecules):
            if i==0:
                datacollector_dict["Molecule 0 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_0(Cell)
            elif i==1:
                datacollector_dict["Molecule 1 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_1(Cell)
            elif i==2:
                datacollector_dict["Molecule 2 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_2(Cell)
            elif i==3:
                datacollector_dict["Molecule 3 exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_3(Cell)
        if self.nb_output_molecules > 1:
            datacollector_dict["Total molecules exchanged"] = lambda m:m.schedule.get_nb_molecules_exchanged_tot(Cell)
        self.datacollector = DataCollector(datacollector_dict)

        # Create cells on the whole grid
        self.schedule.generate_cells(Cell)

        self.running = True
        if fitness_evaluation==False:
            self.datacollector.collect(self)

        # Starting dissimilarity
        self.start_dissimilarity = self.schedule.dissimilarity()


    # Don't delete, or you won't be able to "step" in the visualization in mesa....
    def step(self):
        if self.schedule.time == 100:
            g =     [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [ 1, 1, 1, 1, 1, 3, 3, 1, 1],
                    [ 1, 2, 1, 1, 1, 1, 3, 1, 1],
                    [ 1, 2, 1, 1, 2, 1, 1, 1, 1],
                    [ 1, 2, 1, 1, 1, 1, 1, 1, 1],
                    [ 1, 2, 1, 1, 2, 1, 1, 1, 1],
                    [ 1, 2, 1, 1, 1, 1, 3, 1, 1],
                    [ 1, 1, 1, 1, 1, 3, 3, 1, 1],
                    [ 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            self.goal = g[::-1]
            b =     [[ -80, -80, -80, -80, -80, -80, -80, -80, -80],
                    [ -80, -80, -80, -80, -80, -10, -10, -80, -80],
                    [ -80, -50, -80, -80, -80, -80, -10, -80, -80],
                    [ -80, -50, -80, -80, -50, -80, -80, -80, -80],
                    [ -80, -50, -80, -80, -80, -80, -80, -80, -80],
                    [ -80, -50, -80, -80, -50, -80, -80, -80, -80],
                    [ -80, -50, -80, -80, -80, -80, -10, -80, -80],
                    [ -80, -80, -80, -80, -80, -10, -10, -80, -80],
                    [ -80, -80, -80, -80, -80, -80, -80, -80, -80]]
            self.bioelectric_stimulus = b[::-1]
            self.schedule.reset_energy(self.energy)
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
        # return self.schedule.global_fitness_3()

                

            
 
  
