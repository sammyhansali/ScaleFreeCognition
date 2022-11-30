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
    goal = None
    GJ_opening_molecs = None
    GJ_opening_stress = None
    energy = []
    state = []
    stress = []
    # Directionality
    direction = None
    local_fitness = None
    global_fitness = None
    molecules = None



    def __init__(self, net, depth, unique_id, pos, model, moore, molecules, goal, GJ_opening_molecs, GJ_opening_stress, energy, stress, state, direction, local_fitness, global_fitness):
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
        self.goal = goal
        # Channel Openings
        self.GJ_opening_molecs = GJ_opening_molecs # single variable for now, not sure if should be a list. One ele for each molec type?
        self.GJ_opening_stress = GJ_opening_stress
        # Inputs
        self.energy = energy
        self.state = state
        self.stress = stress
        self.local_fitness = local_fitness
        self.global_fitness = global_fitness
        # Directionality
        self.direction = direction
        
    def net_inputs(self):
        inputs = []
        
        # Determining what you want as inputs
        if "molecules" in self.model.ANN_inputs:
            for x in range(len(self.molecules)):
                inputs.extend(self.molecules[x])
            # inputs.extend(list(self.molecules)) # This will cause an issue!
            # inputs = sum(inputs, [])
        if "goal" in self.model.ANN_inputs:
            inputs.append(self.goal)
        if "energy" in self.model.ANN_inputs:
            inputs.extend(self.energy)
        if "stress" in self.model.ANN_inputs:
            inputs.extend(self.stress)
        if "state" in self.model.ANN_inputs:
            inputs.extend(self.state)
        
        # if "finite_reservoir" in self.model.ANN_inputs:
        #     inputs.append(self.model.depleted_reservoir/self.model.full_reservoir) 
        # if "local_geometrical_frustration" in self.model.ANN_inputs:
        #     inputs.append(self.local_geometrical_frustration()) 
        
        if "local_fitness" in self.model.ANN_inputs:
            inputs.extend(self.local_fitness)
        if "global_fitness" in self.model.ANN_inputs:
            inputs.extend(self.global_fitness)

        # Raw positional data
        if "pos_x" in self.model.ANN_inputs:
            inputs.append(self.pos[0])
        if "pos_y" in self.model.ANN_inputs:
            inputs.append(self.pos[1])
        # Directional history
        if "direction" in self.model.ANN_inputs:
            inputs.extend(self.direction)

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
    

    def update_history(self, var, update):
        return [update] + var[:-1]


    def update_state(self):
        tot = 0
        n = len(self.molecules)
        for i in range(n):
            tot+=self.molecules[i][0]
        
        if tot >=  10*n :
            self.state = self.update_history(self.state, 3)
           
        elif tot >= 5*n:
            self.state = self.update_history(self.state, 2)

        elif tot >= 0:
            self.state = self.update_history(self.state, 1)


    def update_local_fitness(self):
        # new_fit = math.floor(self.model.schedule.local_fitness(self)/100)
        new_fit = self.model.schedule.local_fitness(self)/100
        self.local_fitness = self.update_history(self.local_fitness, new_fit)


    def update_global_fitness(self):
        new_fit = self.model.schedule.global_fitness()/100
        self.global_fitness = self.update_history(self.global_fitness, new_fit)


    def update_direction(self, new_dir):
        self.direction = self.update_history(self.direction, new_dir)


    def prune_stress(self, stress):
        if stress > 100:
            stress = 100
        if stress < 0:
            stress=0     
        return stress
            

    # Experimental
    # def send_molecs_and_stress(self, outputs):
        
    #     neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        
    #     # Stress + anxiolytic distribution
    #     for neighbour in neighbours:
    #         if self.model.grid.is_cell_empty(neighbour)==False: # not dead

    #             cell_in_contact = self.model.grid[neighbour][0]

    #             # Send molecs and update states
    #             self.send_molecs(cell_in_contact)
    #             cell_in_contact.update_state()  
    #             self.update_state()

    #             # stress to send
    #             self.send_stress(cell_in_contact)

    #     if self.GJ_opening_stress>0:
    #         new_stress_amount = self.stress[0] + outputs['stress_to_send'] - outputs['anxio_to_send'] 
    #         self.stress = self.update_history( self.stress, 
    #                         self.prune_stress(new_stress_amount)
    #                         )


    def update_molecs_and_state(self, outputs):
        # Distribute molecules
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        new_self_molecs=[self.molecules[x][0] for x in range(self.model.nb_output_molecules)]

        for neighbour in neighbours:

            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                cic = self.model.grid[neighbour][0] # Cell In Contact
                GJ_open_percentage = min(self.GJ_opening_molecs, cic.GJ_opening_molecs) # For now, assuming only 1 gap junction for all molecs. Not 1 each.

                for i in range(self.model.nb_output_molecules):
                    key = f"molecule_{i}_to_send"
                    new_cic_molecs = 0

                    if new_self_molecs[i] >= outputs[key] * GJ_open_percentage:
                        to_send = outputs[key] * GJ_open_percentage
                        new_cic_molecs = cic.molecules[i][0] + to_send
                        new_self_molecs[i] -= to_send
                    else:
                        new_cic_molecs = cic.molecules[i][0] + new_self_molecs[i]
                        new_self_molecs[i] = 0
                    # Updating histories
                    cic.molecules[i] = cic.update_history(cic.molecules[i], new_cic_molecs)
                    cic.update_state()

        for i in range(self.model.nb_output_molecules):
            self.molecules[i] = self.update_history(self.molecules[i], new_self_molecs[i])
        self.update_state()


    def update_stress(self, outputs):
        # Distribute stress and anxiolytics
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        delta = outputs['stress_to_send'] - outputs['anxio_to_send']
        # Update neighbors
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead

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
            key = f"molecules_{i}_to_send"
            if key in outputs:
                if outputs[key] < 0:
                    outputs[key] = 0
        
        if "stress_to_send" in outputs:
            if outputs["stress_to_send"] < 0:
                outputs["stress_to_send"] = 0

        if "anxio_to_send" in outputs:
            if outputs["anxio_to_send"] < 0:
                outputs["anxio_to_send"] = 0
        
        # molecules1 GJ opening correction
        if "GJ_opening_molecs" in outputs:
            self.GJ_opening_molecs = outputs['GJ_opening_molecs']
            if self.GJ_opening_molecs < 0:
                self.GJ_opening_molecs = 0
            elif self.GJ_opening_molecs > 1:
                self.GJ_opening_molecs = 1

        # stress GJ opening correction
        if "GJ_opening_stress" in outputs:
            self.GJ_opening_stress = outputs['GJ_opening_stress']
            if self.GJ_opening_stress < 0:
                self.GJ_opening_stress = 0
            elif self.GJ_opening_stress > 1:
                self.GJ_opening_stress = 1           

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

        # if "use_finite_reservoir" in outputs:
        #     if outputs["use_finite_reservoir"] < 0:
        #         outputs["use_finite_reservoir"] = 0
        #     elif outputs["use_finite_reservoir"] > 1:
        #         outputs["use_finite_reservoir"] = 1   
        #     outputs["use_finite_reservoir"] = round(outputs["use_finite_reservoir"]) # Want binary
        
        # if "reward" in outputs:
        #     if outputs["reward"] < -1:
        #         # print("went super neg")
        #         outputs["reward"] = -1
        #     elif outputs["reward"] > 1:
        #         # print("went super pos")
        #         outputs["reward"] = 1  
        
        if "direction" in outputs:
            if outputs["direction"] < 0:
                outputs["direction"] = 0
            elif outputs["direction"] > 1:
                outputs["direction"] = 1   
            outputs["direction"] = math.ceil(outputs["direction"]*8) # should be integers between 0 and 8

        return outputs
                
    


    def apoptosis(self, probability):
        if (probability == 1):
            self.die()

    # def cell_division(self, cd_prob):
    def cell_division(self, cd_prob):
        if cd_prob == 1:
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
                    self.divide((x,y), False)

            # Divide in self.direction
    def out_of_bounds(self, pos):
        x,y = pos
        return x < 0 or x >= self.model.width or y < 0 or y >= self.model.height

    def divide(self, pos, subtract):
        x,y = pos

        # Updating initial cell
        # 1. Should give half of its molecs to daughter?
        # 2. Should lose 25 energy? Daughter should always have 50 energy?
        # 3. Should give away half of stress?
        cell = Cell(    net = self.net, 
                        depth = self.depth, 
                        unique_id = self.model.next_id(), 
                        pos = (x,y), 
                        model = self.model,  
                        moore = True, 
                        goal = self.model.goal[y][x], 
                        GJ_opening_molecs = self.GJ_opening_molecs, 
                        GJ_opening_stress = self.GJ_opening_stress, 
                        # Historical data (should be same, but need to update the first indices)
                        energy = self.energy,
                        stress = self.stress, 
                        state = self.state, 
                        local_fitness = self.local_fitness,
                        global_fitness = self.global_fitness,
                        direction = self.direction,
                        molecules = self.molecules, 
                    )
        cell.energy[0] = self.model.energy
        self.birth(cell, x, y, subtract)
        cell.local_fitness[0] = self.model.schedule.local_fitness(self)
        

    def step(self):
        """
        A model step. 
        """
        
        inputs = self.net_inputs()
        outputs = self.prune_outputs(self.net_outputs(inputs))

        # Can only divide when you stop changing directions. Keep this???
        if self.direction[0]==self.direction[1]: 
            self.cell_division(outputs["cell_division"])
        
        if "apoptosis" in outputs:
            self.apoptosis(outputs["apoptosis"])

        # Updating variables
        self.energy = self.update_history(self.energy, self.energy[0]+self.global_fitness[0]-0.1)
        self.update_global_fitness()
        # self.update_local_fitness()
        self.update_direction(outputs["direction"])
        self.update_molecs_and_state(outputs)
        self.update_stress(outputs)

        if self.energy[0] <= 0:
            self.die()
            return
        

    def die(self):
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)

    def birth(self, cell, x, y, subtract): 
        self.model.grid.place_agent(cell, (x, y))
        self.model.schedule.add(cell)

