from mesa import Agent
import random
from enum import IntEnum
import numpy as np
import sys
import multicellularity.schedule
import math


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
    x = None
    y = None
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



    def __init__(self, net, depth, unique_id, pos, model, moore, molecules, goal_cell_type, bioelectric_stimulus, GJ_opening_ions, GJ_molecules, GJ_opening_stress, energy, stress, direction, global_fitness, cell_type, potential):
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
        inputs = []
        
        # Determining what you want as inputs
        if "molecules" in self.model.ANN_inputs:
            for x in range(len(self.molecules)):
                inputs.extend(self.molecules[x])
        if "goal_cell_type" in self.model.ANN_inputs:
            inputs.append(self.goal_cell_type)
        if "bioelectric_stimulus" in self.model.ANN_inputs:
            inputs.append(self.bioelectric_stimulus)
        if "energy" in self.model.ANN_inputs:
            inputs.extend(self.energy)
        if "stress" in self.model.ANN_inputs:
            inputs.extend(self.stress)
        if "state" in self.model.ANN_inputs:
            inputs.extend(self.state)
        if "cell_type" in self.model.ANN_inputs:
            inputs.extend(self.cell_type)
        if "potential" in self.model.ANN_inputs:
            inputs.extend(self.potential)

        # Fitness histories        
        if "global_fitness" in self.model.ANN_inputs:
            inputs.extend(self.global_fitness)

        # Directional history
        if "direction" in self.model.ANN_inputs:
            inputs.extend(self.direction)

        # Raw positional data
        if "pos_x" in self.model.ANN_inputs:
            inputs.append(self.pos[0])
        if "pos_y" in self.model.ANN_inputs:
            inputs.append(self.pos[1])

        # Bias
        if "bias" in self.model.ANN_inputs:
            inputs.append(0.5)
        
        return(inputs)
    

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


    def update_global_fitness(self):
        new_fit = self.model.schedule.global_fitness()/100
        # new_fit = self.model.schedule.global_fitness_3()/100
        self.global_fitness = self.update_history(self.global_fitness, new_fit)


    def update_direction(self, new_dir):
        self.direction = self.update_history(self.direction, new_dir)


    def update_cell_type(self):
        # Get the total number of molecules
        n = len(self.molecules)
        tot = sum(self.molecules[i][0] for i in range(n))
        
        if tot >=  10*n :
            self.cell_type = self.update_history(self.cell_type, 3)
           
        elif tot >= 5*n:
            self.cell_type = self.update_history(self.cell_type, 2)

        elif tot >= 0:
            self.cell_type = self.update_history(self.cell_type, 1)

        # E penalty for cell_type change
        if self.cell_type[0] != self.cell_type[1]:
            self.energy_temp -= 0.5
            
    def neighbour_is_alive(self, neighbour):
        return self.model.grid.is_cell_empty(neighbour) == False

    # Membrane potential
    # Made this function. This obviously utilizes GJ open percentage.
    def update_potential(self, charge_to_send):
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        new_self_potential = self.potential[0]

        for neighbour in neighbours:

            if self.neighbour_is_alive(neighbour):
                cic = self.model.grid[neighbour][0] # Cell In Contact
                GJ_ions = min(self.GJ_opening_ions, cic.GJ_opening_ions)

                new_cic_potential = 0
                if new_self_potential >= charge_to_send * GJ_ions:
                    to_send = charge_to_send * GJ_ions
                    # Cells cannot have less than -100 charge
                    if cic.potential[0] + to_send > -100: 
                        new_cic_potential = cic.potential[0] + to_send
                        new_self_potential -= to_send
                        # Updating history
                        cic.potential = cic.update_history(cic.potential, new_cic_potential)
                else:
                    # Cells cannot have less than -100 charge
                    if cic.potential[0] + new_self_potential > -100:
                        new_cic_potential = cic.potential[0] + new_self_potential
                        new_self_potential = 0
                        # Update and end early
                        cic.potential = cic.update_history(cic.potential, new_cic_potential)
                        break
        # Updating self history
        self.potential = self.update_history(self.potential, new_self_potential)

    # def update_molecs_and_state(self, outputs):
    # Changed this so it no longer updates state this that has no meaning anymore.
    # Also, not sure whether to keep using GJ opening...
    # Now, this is supposed to represent signaling molecs - not ions. Can be produced arbitrarily or not.
    # Now it also makes more sense when you increase amount of molecs...youre increasing numbers of signals available.
    def update_molecs(self, outputs):

        new_self_molecs=[self.molecules[x][0] for x in range(self.model.nb_output_molecules)]

        for neighbour in self.model.grid.get_neighborhood(self.pos, self.moore, False):

            if self.neighbour_is_alive(neighbour):
                cic = self.model.grid[neighbour][0] # Cell In Contact
                new_self_molecs = self.update_molecs_neighbor(cic, new_self_molecs, outputs)

        for i in range(self.model.nb_output_molecules):
            self.molecules[i] = self.update_history(self.molecules[i], new_self_molecs[i])
        self.update_cell_type()

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
        cic.update_cell_type()
        return new_self_molecs


    def update_stress(self, stress_to_send, anxio_to_send):
        # Distribute stress and anxiolytics
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        delta = stress_to_send - anxio_to_send
        # Update neighbors
        for neighbour in neighbours:
            if self.neighbour_is_alive(neighbour):

                cic = self.model.grid[neighbour][0] # Cell In Contact
                GJ_open_percentage = self.GJ_opening_stress * cic.GJ_opening_stress # Change to min?
                new_cic_stress = cic.stress[0] + GJ_open_percentage * delta
                cic.stress = cic.update_history    (   cic.stress, 
                                                        cic.prune_stress(new_cic_stress)
                                                    )
        
        # Update self
        if self.GJ_opening_stress>0:
            new_self_stress =   self.stress[0] + delta 
            self.stress = self.update_history   (   self.stress, 
                                                    self.prune_stress(new_self_stress)
                                                )


    def prune_outputs(self, outputs):
        """Make sure outputs are in correct form for use with agent actions"""
                    
        # Multi molecule support
        for i in range(self.model.nb_output_molecules):
            key = f"molecule_{i}_to_send"
            if key in outputs:
                if outputs[key] < 0:
                    outputs[key] = 0

            key = f"molecule_{i}_GJ"
            if key in outputs:
                if outputs[key] < 0:
                    outputs[key] = 0
                elif outputs[key] > 1:
                    outputs[key] = 1
                self.GJ_molecules[i] = outputs[key]
        
        if "stress_to_send" in outputs:
            if outputs["stress_to_send"] < 0:
                outputs["stress_to_send"] = 0

        if "anxio_to_send" in outputs:
            if outputs["anxio_to_send"] < 0:
                outputs["anxio_to_send"] = 0
        
        # ions GJ opening correction
        if "GJ_opening_ions" in outputs:
            if outputs['GJ_opening_ions'] < 0:
                outputs['GJ_opening_ions'] = 0
            elif outputs['GJ_opening_ions'] > 1:
                outputs['GJ_opening_ions'] = 1
            self.GJ_opening_ions = outputs['GJ_opening_ions']

        # # molecules GJ opening correction
        # if "GJ_opening_molecs" in outputs:
        #     if outputs['GJ_opening_molecs'] < 0:
        #         outputs['GJ_opening_molecs'] = 0
        #     elif outputs['GJ_opening_molecs'] > 1:
        #         outputs['GJ_opening_molecs'] = 1
        #     self.GJ_opening_molecs = outputs['GJ_opening_molecs']

        # stress GJ opening correction
        if "GJ_opening_stress" in outputs:
            if outputs['GJ_opening_stress'] < 0:
                outputs['GJ_opening_stress'] = 0
            elif outputs['GJ_opening_stress'] > 1:
                outputs['GJ_opening_stress'] = 1        
            self.GJ_opening_stress = outputs['GJ_opening_stress']

        # make apoptosis variable binary
        if "apoptosis" in outputs:
            if outputs["apoptosis"] < 0:
                outputs["apoptosis"] = 0
            elif outputs["apoptosis"] > 1:
                outputs["apoptosis"] = 1   
            outputs["apoptosis"] = round(outputs["apoptosis"]) # Want binary

        # make cell div variable binary
        if "cell_division" in outputs:
            if outputs["cell_division"] < 0:
                outputs["cell_division"] = 0
            elif outputs["cell_division"] > 1:
                outputs["cell_division"] = 1
            outputs["cell_division"] = round(outputs["cell_division"]) # Want binary

        if "direction" in outputs:
            if outputs["direction"] < 0:
                outputs["direction"] = 0
            elif outputs["direction"] > 1:
                outputs["direction"] = 1   
            outputs["direction"] = math.ceil(outputs["direction"]*8) # should be integers between 1 and 8

        if "cell_type" in outputs:
            if outputs["cell_type"] < 0:
                outputs["cell_type"] = 0
            elif outputs["cell_type"] > 1:
                outputs["cell_type"] = 1   
            outputs["cell_type"] = math.ceil(outputs["cell_type"]*3) # should be integers between 1 and 3

        return outputs
                
    def differentiate(self, cell_type):
        self.cell_type = self.update_history(self.cell_type, cell_type)

        # Transition penalty
        if self.cell_type[0] != self.cell_type[1]:
            self.energy_temp -= 0.5

    # def cell_division(self, cd_prob):
    def cell_division(self):
        x = self.pos[0]
        y = self.pos[1]
        # Check if direction is on the grid and empty 
        # (technically could still divide if full, and would have to push the existing cell.
        # don't think can implement this unless we use something without 2D grid limits.)
        if self.direction[0] == 1:     # NO
            y+=1
        elif self.direction[0] == 2:   # NE
            x+=1
            y+=1
        elif self.direction[0] == 3:   # EA
            x+=1
        elif self.direction[0] == 4:   # SE
            x+=1
            y-=1
        elif self.direction[0] == 5:   # SO
            y-=1
        elif self.direction[0] == 6:   # SW
            x-=1
            y-=1
        elif self.direction[0] == 7:   # WE
            x-=1
        elif self.direction[0] == 8:   # NW
            x-=1
            y+=1
        
        if not self.out_of_bounds((x,y)):
            if len(self.model.grid.get_cell_list_contents([(x,y)]))==0:
                self.divide((x,y))

        
    def out_of_bounds(self, pos):
        x,y = pos
        return x < 0 or x >= self.model.width or y < 0 or y >= self.model.height

    def divide(self, pos):
        x,y = pos

        # Updating initial cell
        self.energy_temp -= 0
        # 1. Should give half of its molecs to daughter?
        # 2. Should lose 1 energy? Daughter should always have 50 energy?
        # 3. Should give away half of stress?
        cell = Cell(    net = self.net, 
                        depth = self.depth, 
                        unique_id = self.model.next_id(), 
                        pos = (x,y), 
                        model = self.model,  
                        moore = True, 
                        goal_cell_type = self.model.goal[y][x], 
                        GJ_opening_ions = self.GJ_opening_ions, 
                        GJ_molecules = self.GJ_molecules, 
                        GJ_opening_stress = self.GJ_opening_stress, 
                        # Historical data (should be same, but need to update the first indices)
                        energy = self.energy,
                        stress = self.stress, 
                        global_fitness = self.global_fitness,
                        direction = self.direction,
                        molecules = self.molecules, 
                    )
        cell.energy =  self.update_history(cell.energy, self.model.energy)
        self.birth(cell, x, y)
        # cell.local_fitness[0] = self.model.schedule.local_fitness(self)


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

        # Differentiate
        if "cell_type" in outputs:
            self.differentiate(outputs["cell_type"])

        
        # Death
        if "apoptosis" in outputs:
            if outputs["apoptosis"] == 1:
                self.die()
                return

        if self.energy[0] <= 0:
            self.die()
            return

        # # Test
        # if self.molecules[0][0] > 10:
        #     self.die()
        #     return

        # Updating variables
        if "global_fitness" in self.model.ANN_inputs:
            self.update_global_fitness()
        if "direction" in self.model.ANN_inputs:
            self.update_direction(outputs["direction"])   
        if "molecules" in self.model.ANN_inputs:
            self.update_molecs(outputs)
        if "potential" in self.model.ANN_inputs:
            if "charge_to_send" in self.model.ANN_outputs:
                self.update_potential(outputs["charge_to_send"])
            else:
                self.potential = self.update_history(self.potential, self.potential[0])
        if "stress" in self.model.ANN_inputs:
            self.update_stress(outputs["stress_to_send"], outputs["anxio_to_send"])
        # E needs to be updated last since the above updates can sap E
        if "energy" in self.model.ANN_inputs:
            self.energy = self.update_history(self.energy, self.energy_temp+self.global_fitness[0]-self.model.e_penalty)

        # if self.global_fitness[0] < 0.95:
        #   self.energy = self.update_history(self.energy, self.energy_temp - 0.6) # So dies in less than 100 turns
        # else:
        #   self.energy = self.update_history(self.energy, self.energy_temp + 0.6)
        
        

    def die(self):
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)

    def birth(self, cell, x, y): 
        self.model.grid.place_agent(cell, (x, y))
        self.model.schedule.add(cell)

