from mesa import Agent
import random
from enum import IntEnum
import sys

sys.setrecursionlimit(10000)

limit = 1
multiplier = 20
stress_unity=2.5


class State(IntEnum):
    NEUTRAL = 0
    POLARIZED = 1
    DEPOLARIZED = 2

class Cell(Agent):
    """
    A cell that walks according to the outputs of an evolved neural network, and eat food to stay alive.
    """
    
    net=None
    depth = None
    grid = None
    pos_x = None
    pos_y = None
    moore = True   
    goal_cell_type = None
    bioelectric_stimulus = None
    GJ_opening_ions = None
    GJ_molecules = None
    GJ_opening_stress = None
    energy = []
    state = []
    stress = []
    # Directionality
    direction = None
    global_fitness = None
    molecules = None
    energy_temp = None
    cell_type = None
    potential = None
    bias = None



    def __init__(self, net, depth, unique_id, pos, model, moore, molecules, goal_cell_type, global_fitness, bioelectric_stimulus, GJ_opening_ions, GJ_molecules, GJ_opening_stress, energy, stress, direction, cell_type, potential):
        """
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        moore: If True, may move in all 8 directions.
                Otherwise, only up, down, left, right.
        """
        super().__init__(unique_id, model)
        self.net=net
        self.depth=depth
        self.pos = pos
        self.pos_x = self.pos[0]
        self.pos_y = self.pos[1]
        self.moore = moore
        self.molecules = molecules # dict, one history list for each molec type
        self.goal_cell_type = goal_cell_type
        self.bioelectric_stimulus = bioelectric_stimulus
        # Channel Openings
        self.GJ_opening_ions = GJ_opening_ions # single variable for now, not sure if should be a list. One ele for each molec type?
        self.GJ_molecules = GJ_molecules
        self.GJ_opening_stress = GJ_opening_stress
        # Inputs
        self.energy = energy
        self.stress = stress
        self.global_fitness = global_fitness
        # Directionality
        self.direction = direction
        self.energy_temp = self.energy[0]
        self.cell_type = cell_type
        self.potential = potential
        
    def net_inputs(self):
        # ### Code that doesn't work, but should do the same
        # inputs = [self.__getattribute__(inp_name) for inp_name in set(self.model.ANN_inputs) if self.__getattribute__(inp_name) is not None]
        # # Flatten the list of lists
        # flattened_inputs = []
        # for item in inputs:
        #     if isinstance(item, dict):
        #         for key, value in item.items():
        #             flattened_inputs.extend(value)
        #     elif isinstance(item, (list,tuple)):
        #         flattened_inputs.extend(item)
        #     else:
        #         flattened_inputs.append(item)

        # # Bias of 0.5
        # flattened_inputs.append(0.5)
        # return flattened_inputs

        ### Code that works
        inputs = []
        if "molecules" in self.model.ANN_inputs:
            for x in range(len(self.molecules)):
                inputs.extend(self.molecules[x])
        if "energy" in self.model.ANN_inputs:
            inputs.extend(self.energy)
        if "cell_type" in self.model.ANN_inputs:
            inputs.extend(self.cell_type)
        if "potential" in self.model.ANN_inputs:
            inputs.extend(self.potential)
        # Bias of 0.5
        inputs.append(0.5)
        return inputs

    def net_outputs(self,new_input):
        #outputs network    
        self.net.Flush()
        self.net.Input(new_input)
        [self.net.Activate() for _ in range(self.depth)]
        raw_outputs = list(self.net.Output())
        outputs = {k: v for k,v in zip(self.model.ANN_outputs, raw_outputs)}

        return outputs
    
    def prune_stress(self, stress):
        if stress > 100:
            stress = 100
        if stress < 0:
            stress=0     
        return stress

    def update_history(self, var, update):
        return [update] + var[:-1]


    def update_direction(self, new_dir):
        self.direction = self.update_history(self.direction, new_dir)


    def update_cell_type_with_molecs(self):
        # Get the total number of molecules
        n = len(self.molecules)
        tot = sum(self.molecules[i][0] for i in range(n))
        multiple = self.model.multiple # 5 seems to work for 3 cell types. for 4 idk yet.

        # Only multiple changes, not the leading coefficients.
        # if tot >  3*multiple*n :
        #     self.cell_type = self.update_history(self.cell_type, 4)

        if tot >=  2*multiple*n :
            self.cell_type = self.update_history(self.cell_type, 3)
           
        elif tot >= multiple*n:
            self.cell_type = self.update_history(self.cell_type, 2)

        elif tot >= 0:
            self.cell_type = self.update_history(self.cell_type, 1)

        # E penalty for cell_type change
        if self.cell_type[0] != self.cell_type[1]:
            self.energy_temp -= 0.5
            
    def neighbour_is_alive(self, neighbour):
        return self.model.grid.is_cell_empty(neighbour) == False


    def update_molecs(self, outputs):

        new_self_molecs=[self.molecules[x][0] for x in range(self.model.nb_output_molecules)]
        neighborhood = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        random.shuffle(neighborhood)
        for neighbour in neighborhood:

            if self.neighbour_is_alive(neighbour):
                cic = self.model.grid[neighbour][0] # Cell In Contact
                new_self_molecs = self.update_molecs_neighbor(cic, new_self_molecs, outputs)

        for i in range(self.model.nb_output_molecules):
            self.molecules[i] = self.update_history(self.molecules[i], new_self_molecs[i])

    def update_molecs_neighbor(self, cic, new_self_molecs, outputs):
        for i in range(self.model.nb_output_molecules):
            GJ_molecules = min(self.GJ_molecules[i], cic.GJ_molecules[i])
            new_cic_molecs = 0

            key = f"molecule_{i}_to_send"
            value = outputs[key]*GJ_molecules

            if new_self_molecs[i] >= value:
                new_cic_molecs = cic.molecules[i][0] + value
                new_self_molecs[i] -= value
            else:
                new_cic_molecs = cic.molecules[i][0] + new_self_molecs[i]
                new_self_molecs[i] = 0
            # Updating histories
            cic.molecules[i] = cic.update_history(cic.molecules[i], new_cic_molecs)
        # cic.update_cell_type()
        return new_self_molecs


    def prune_outputs(self, outputs):
        """Make sure outputs are in correct form for use with agent actions"""

        # Multi molecule support
        molecules_range = [("molecule_{}_to_send", (0, float('inf'))), ("molecule_{}_GJ", (0, 1))]
        for i in range(self.model.nb_output_molecules):
            for key, (low, high) in molecules_range:
                key = key.format(i)
                if key in outputs:
                    if outputs[key] < low:
                        outputs[key] = low
                    elif outputs[key] > high:
                        outputs[key] = high
                    if key.endswith("_GJ"):
                        self.GJ_molecules[i] = outputs[key]

        ranges = {
            # "apoptosis": (0, 1),
            # "cell_division": (0, 1),
            # "direction": (0, 1),
            "differentiate": (0, 1)
        }

        for key, (low, high) in ranges.items():
            if key in outputs:
                if outputs[key] < low:
                    outputs[key] = low
                elif outputs[key] > high:
                    outputs[key] = high

                # Specifics
                if key == "direction":
                    outputs[key] = math.ceil(outputs[key]*8) # should be integers between 1 and 8
                if key == "differentiate":
                    outputs[key] = math.ceil(outputs[key]*4) # should be integers between 1 and 4
                if key == "GJ_opening_ions":
                    self.GJ_opening_ions = outputs[key]
                if key == "apoptosis":
                    outputs[key] = round(outputs[key])
                if key == "cell_division":
                    outputs[key] = round(outputs[key])

        return outputs
                
    def differentiate(self, cell_type):
        self.cell_type = self.update_history(self.cell_type, cell_type)

        # Transition penalty
        if self.cell_type[0] != self.cell_type[1]:
            self.energy_temp -= 0.5

    # def cell_division(self, cd_prob):
    # def cell_division(self):
    #     x = self.pos[0]
    #     y = self.pos[1]
    #     # Check if direction is on the grid and empty 
    #     # (technically could still divide if full, and would have to push the existing cell.
    #     # don't think can implement this unless we use something without 2D grid limits.)
    #     if self.direction[0] == 1:     # NO
    #         y+=1
    #     elif self.direction[0] == 2:   # NE
    #         x+=1
    #         y+=1
    #     elif self.direction[0] == 3:   # EA
    #         x+=1
    #     elif self.direction[0] == 4:   # SE
    #         x+=1
    #         y-=1
    #     elif self.direction[0] == 5:   # SO
    #         y-=1
    #     elif self.direction[0] == 6:   # SW
    #         x-=1
    #         y-=1
    #     elif self.direction[0] == 7:   # WE
    #         x-=1
    #     elif self.direction[0] == 8:   # NW
    #         x-=1
    #         y+=1
        
    #     if not self.out_of_bounds((x,y)):
    #         if len(self.model.grid.get_cell_list_contents([(x,y)]))==0:
    #             self.divide((x,y))

        
    # def out_of_bounds(self, pos):
    #     x,y = pos
    #     return x < 0 or x >= self.model.width or y < 0 or y >= self.model.height

    # def divide(self, pos):
    #     x,y = pos

    #     # Updating initial cell
    #     self.energy_temp -= 0
    #     # 1. Should give half of its molecs to daughter?
    #     # 2. Should lose 1 energy? Daughter should always have 50 energy?
    #     # 3. Should give away half of stress?
    #     cell = Cell(    net = self.net, 
    #                     depth = self.depth, 
    #                     unique_id = self.model.next_id(), 
    #                     pos = (x,y), 
    #                     model = self.model,  
    #                     moore = True, 
    #                     goal_cell_type = self.model.goal[y][x], 
    #                     GJ_opening_ions = self.GJ_opening_ions, 
    #                     GJ_molecules = self.GJ_molecules, 
    #                     GJ_opening_stress = self.GJ_opening_stress, 
    #                     # Historical data (should be same, but need to update the first indices)
    #                     energy = self.energy,
    #                     stress = self.stress, 
    #                     global_fitness = self.global_fitness,
    #                     direction = self.direction,
    #                     molecules = self.molecules, 
    #                 )
    #     cell.energy =  self.update_history(cell.energy, self.model.energy)
    #     self.birth(cell, x, y)
    #     # cell.local_fitness[0] = self.model.schedule.local_fitness(self)


    def step(self):
        """
        A model step. 
        """
        
        inputs = self.net_inputs()
        outputs = self.prune_outputs(self.net_outputs(inputs))
        self.energy_temp = self.energy[0]

        # Division
        if "cell_division" in outputs:
            if self.energy_temp >= 0 and outputs["cell_division"] == 1:
                self.cell_division()

        # Differentiation
        if "differentiate" in outputs:
            self.differentiate(outputs["differentiate"])

        # Death
        if "apoptosis" in outputs:
            if outputs["apoptosis"] == 1:
                self.die()
                return
        if self.energy[0] <= 0:
            self.die()
            return

        # # Updating variables
        self.global_fitness = self.model.global_fitness

        # if "direction" in self.model.ANN_inputs:
        #     self.update_direction(outputs["direction"])   
        if "molecules" in self.model.ANN_inputs:
            self.update_molecs(outputs)
            if not "differentiate" in outputs:
                self.update_cell_type_with_molecs()

        if "potential" in self.model.ANN_inputs:
            self.potential = self.update_history(self.potential, self.model.bioelectric_stimulus[self.pos[1]][self.pos[0]])

        # if "stress" in self.model.ANN_inputs:
        #     self.update_stress(outputs["stress_to_send"], outputs["anxio_to_send"])
        # # E needs to be updated last since the above updates can sap E
        if "energy" in self.model.ANN_inputs:
            # self.model.update_global_fitness()
            self.energy = self.update_history(self.energy, self.energy_temp+self.model.global_fitness[0]-self.model.e_penalty)

    def die(self):
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)

    def birth(self, cell, x, y): 
        self.model.grid.place_agent(cell, (x, y))
        self.model.schedule.add(cell)

