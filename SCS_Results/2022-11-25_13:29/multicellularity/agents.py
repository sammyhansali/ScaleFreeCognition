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
    # Directionality
    NO=1
    NE=2
    EA=3
    SE=4
    SO=5
    SW=6
    WE=7
    NW=8

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
    molecules = None
    goal=None
    GJ_opening_molecs = None
    GJ_opening_stress = None
    energy = []
    state = []
    stress = []
    # Directionality
    direction = None
    # How about an energy list? energy[0] is cells energy at t=x. energy[1] is energy at t=x-1. Etc...
    # Similar idea with stress and state and molecules
    # And of course with tissue state


    # def __init__(self, net, depth, unique_id, pos, model, moore, molecules, energy, energyt1,  goal, GJ_opening_molecs, GJ_opening_stress, stress, stresst1, state, statet1):
    def __init__(self, net, depth, unique_id, pos, model, moore, molecules, goal, GJ_opening_molecs, GJ_opening_stress, energy, stress, state, direction):
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
        self.molecules = molecules # list, one for each molec type
        self.goal = goal
        # Channel Openings
        self.GJ_opening_molecs = GJ_opening_molecs # single variable for now, not sure if should be a list. One ele for each molec type?
        self.GJ_opening_stress = GJ_opening_stress
        # Inputs
        self.energy = energy
        self.state = state
        self.stress = stress
        # Directionality
        self.direction = direction
        
    def net_inputs(self):
        inputs = []
        
        # Determining what you want as inputs
        if "molecules" in self.model.ANN_inputs:
            inputs.extend(list(self.molecules))
            # inputs = sum(inputs, [])
        if "energy" in self.model.ANN_inputs:
            inputs.extend(self.energy)
        if "stress" in self.model.ANN_inputs:
            inputs.extend(self.stress)
        if "state" in self.model.ANN_inputs:
            inputs.extend(self.state)
        if "delta" in self.model.ANN_inputs:
            inputs.append(self.goal-self.state[0])
        # if "energy" in self.model.ANN_inputs:
        #     inputs.append(self.energy)
        # if "energyt1" in self.model.ANN_inputs:
        #     inputs.append(self.energyt1)
        # if "stress" in self.model.ANN_inputs:
        #     inputs.append(self.stress)
        # if "stresst1" in self.model.ANN_inputs:
        #     inputs.append(self.stresst1)
        # if "state_goal" in self.model.ANN_inputs:
        #     inputs.append(self.goal)
        # if "state" in self.model.ANN_inputs:
        #     inputs.append(self.state)
        # if "statet1" in self.model.ANN_inputs:
        #     inputs.append(self.statet1) 
        
        # if "finite_reservoir" in self.model.ANN_inputs:
        #     inputs.append(self.model.depleted_reservoir/self.model.full_reservoir) 
        # # N = 1
        # if "local_geometrical_frustration" in self.model.ANN_inputs:
        #     inputs.append(self.local_geometrical_frustration()) 
        if "local_state" in self.model.ANN_inputs:
            print("LS")
            inputs.append(self.local_state()) 
        if "fitness_score" in self.model.ANN_inputs:
            print("FS")
            inputs.append(self.model.schedule.fitness()/100)

        # Raw positional data
        if "pos_x" in self.model.ANN_inputs:
            inputs.append(self.pos[0])
            # inputs.append(self.pos[0]/8)
        if "pos_y" in self.model.ANN_inputs:
            inputs.append(self.pos[1])
            # inputs.append(self.pos[1]/8)
        if "direction" in self.model.ANN_inputs:
            inputs.append(self.direction)

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
        # Might want to try 4 different cell_division outputs
        outputs = {k: v for k,v in zip(self.model.ANN_outputs, raw_outputs)}
        
        return outputs
    
    def update_history(self, var, update):
        return [update] + var[:-1]

    def update_state(self):
        if self.molecules[0] >=  10 :
            self.state = self.update_history(self.state, 1)
            # if doesn't work: self.state = [1] + self.state[:-1]
           
        elif  self.molecules[0] < 10 and self.molecules[0] >= 5:
            self.state = self.update_history(self.state, 3)

        elif self.molecules[0] < 5 and self.molecules[0] >= 0:
            self.state = self.update_history(self.state, 2)
            
    def prune_stress(self, stress):
        if stress > 100:
            stress = 100
        if stress < 0:
            stress=0     
        return stress
            
    def send_ions_and_stress(self, outputs):
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        
        # Molecule 1 distribution
        # for i in range(self.molecules):
        #     key = f"m{i}_to_send"
        i=0                 # placeholder, later will put loop like above
        key="m0_to_send"    # placeholder, later will put loop like above      
        
        # Stress + anxiolytic distribution
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                cell_in_contact = self.model.grid[neighbour][0]
                
                # m0 to send
                GJ_open_percentage = min(self.GJ_opening_molecs, cell_in_contact.GJ_opening_molecs)
                if self.molecules[i] >= outputs[key] * GJ_open_percentage:
                    cell_in_contact.molecules[i] += outputs[key] * GJ_open_percentage   
                    self.molecules[i] -= outputs[key] * GJ_open_percentage   
                else:
                    cell_in_contact.molecules[i] += self.molecules[i]
                    self.molecules[i] = 0
                cell_in_contact.update_state()  

                # stress to send
                if self.GJ_opening_stress>0:
                    opening = self.GJ_opening_stress * cell_in_contact.GJ_opening_stress
                    new_stress_amount = cell_in_contact.stress[0] + outputs['stress_to_send']*(opening) - outputs['anxio_to_send']*(opening)
                    cell_in_contact.stress = self.update_history( cell_in_contact.stress, 
                                    self.prune_stress(new_stress_amount)
                                    )
                    # cell_in_contact.stress += outputs['stress_to_send']*((self.GJ_opening_stress * cell_in_contact.GJ_opening_stress))   
                    # cell_in_contact.stress -= outputs['anxio_to_send']*((self.GJ_opening_stress * cell_in_contact.GJ_opening_stress))   

        if self.GJ_opening_stress>0:
            new_stress_amount = self.stress[0] + outputs['stress_to_send'] - outputs['anxio_to_send'] 
            self.stress = self.update_history( self.stress, 
                            self.prune_stress(new_stress_amount)
                            )

        self.update_state()   


        
    def prune_outputs(self, outputs):
        """Make sure outputs are in correct form for use with agent actions"""
                    
        # supports multiple molecules sent out in future
        # for i in range(self.molecules):
        #     key = f"m{i+1}_to_send"
        key ="m0_to_send"
        if outputs[key] < 0:
            outputs[key] = 0
        if outputs["stress_to_send"] < 0:
            outputs["stress_to_send"] = 0
        if outputs["anxio_to_send"] < 0:
            outputs["anxio_to_send"] = 0
        
        # molecules1 GJ opening correction
        self.GJ_opening_molecs = outputs['GJ_opening_molecs']
        if self.GJ_opening_molecs < 0:
            self.GJ_opening_molecs = 0
        elif self.GJ_opening_molecs > 1:
            self.GJ_opening_molecs = 1

        # stress GJ opening correction
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

        if "use_finite_reservoir" in outputs:
            if outputs["use_finite_reservoir"] < 0:
                outputs["use_finite_reservoir"] = 0
            elif outputs["use_finite_reservoir"] > 1:
                outputs["use_finite_reservoir"] = 1   
            outputs["use_finite_reservoir"] = round(outputs["use_finite_reservoir"]) # Want binary
        
        if "reward" in outputs:
            if outputs["reward"] < -1:
                # print("went super neg")
                outputs["reward"] = -1
            elif outputs["reward"] > 1:
                # print("went super pos")
                outputs["reward"] = 1  
        
        if "direction" in outputs:
            if outputs["direction"] < 0:
                outputs["direction"] = 0
            elif outputs["direction"] > 1:
                outputs["direction"] = 1   
            outputs["direction"] = math.ceil(outputs["direction"]*8) # should be integers between 0 and 8

        return outputs
                
    def local_geometrical_frustration (self):
        geometrical_frustration = 0
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                 cell_in_contact = self.model.grid.get_cell_list_contents([neighbour])[0]
                 if self.state[0] != cell_in_contact.state[0]:
                     geometrical_frustration += 1
        return geometrical_frustration / (len(neighbours))

    # reward by concentric, or by N=1 nearest neighbours
    def local_state(self):
        local_state = 0
        # Put true so score is lower if the center cell is not in correct state.
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, True)
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                cell = self.model.grid.get_cell_list_contents([neighbour])[0]
                if cell.state[0] == cell.goal:
                    local_state += 1
            else:
                # Need to get x and y coords of the dead neighbour
                x=neighbour[0]
                y=neighbour[1]
                if self.model.goal[y][x] == 0:
                    local_state += 1
        return local_state / (len(neighbours))

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
            if self.direction == 1:     # NO
                y+=1
            elif self.direction == 2:   # NE
                x+=1
                y+=1
            elif self.direction == 3:   # EA
                x+=1
            elif self.direction == 4:   # SE
                x+=1
                y-=1
            elif self.direction == 5:   # SO
                y-=1
            elif self.direction == 6:   # SW
                x-=1
                y-=1
            elif self.direction == 7:   # WE
                x-=1
            elif self.direction == 8:   # NW
                x-=1
                y+=1
            
            if not self.out_of_bounds((x,y)):
                if len(self.model.grid.get_cell_list_contents([(x,y)]))==0:
                    self.divide((x,y), False)

            # Divide in self.direction
    def out_of_bounds(self, pos):
        x,y = pos
        return x < 0 or x >= self.model.width or y < 0 or y >= self.model.height

        # n = len(neighbours)
        # # Get discrete numbers 0, 1, 2,..., n
        # index = round( cd_prob * n ) 

        # # Don't divide if index == n
        # if index != n:
        #     # If desired position is empty, divide. Else can't divide.
        #     if self.model.grid.is_cell_empty(neighbours[index])==True:
        #         subtract=False
        #         # if "use_finite_reservoir" in outputs:
        #         #     if outputs["use_finite_reservoir"]==1:
        #         #         subtract=True
        #         self.divide(neighbours[index], subtract)
                

    def divide(self, pos, subtract):
        # x = pos[0]
        # y = pos[1]
        x,y = pos

        # Updating initial cell
        history_length = 2
        cell = Cell(    net = self.net, 
                        depth = self.depth, 
                        unique_id = self.model.next_id(), 
                        pos = (x,y), 
                        model = self.model,  
                        moore = True, 
                        molecules = self.molecules, 
                        goal = self.model.goal[y][x], 
                        GJ_opening_molecs = self.GJ_opening_molecs, 
                        GJ_opening_stress = self.GJ_opening_stress, 
                        # Historical data
                        energy = [0]*history_length,
                        stress = self.stress, # Giving same history of stress
                        state = self.state, # Giving same history of state
                        direction = self.direction,
                    )
        cell.energy[0] = self.model.energy
        self.birth(cell, x, y, subtract)
        

    def step(self):
        """
        A model step. 
        """
        
        # self.energyt1 = self.energy
        # self.stresst1 = self.stress
        # self.statet1 = self.state
                
        inputs = self.net_inputs()
        outputs = self.prune_outputs(self.net_outputs(inputs))
        self.send_ions_and_stress(outputs)

        # Try this out next, cells only survive 100 steps (unless they do apop early)
        # if "reward" in outputs:
        #     self.energy += outputs["reward"] - 1.0
        # else:
        #     self.energy -= 0.5
        # self.energy[0] += 0.5*(self.local_state()) + 0.5*(self.schedule.fitness()/100) - 0.8
        # new_energy = self.energy[0] + 0.5*(self.local_state()) + 0.5*(self.model.schedule.fitness()/100) - 0.8
        # self.energy = self.update_history(self.energy, new_energy)

        if self.energy[0] <= 0:
            self.die()
            return
        
        self.energy = self.update_history(self.energy, self.energy[0])
            
        if "direction" in outputs:
            if outputs["direction"] == self.direction: # so direction not changing
                self.cell_division(outputs["cell_division"])
            else:
                # Take 1 turn to rotate self. Can potentially divide next turn.
                self.direction = outputs["direction"]
        
        
        # if "cell_division" in outputs:
        #         self.cell_division(outputs["cell_division"])

        if "apoptosis" in outputs:
            self.apoptosis(outputs["apoptosis"])
        

    def die(self):
        # self.model.depleted_reservoir += 1    # Testing if increasing reservoir after death makes sense
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)

    def birth(self, cell, x, y, subtract): 
        # if subtract:
            # self.model.depleted_reservoir -= 1
        # self.model.depleted_reservoir -= 1
        self.model.grid.place_agent(cell, (x, y))
        self.model.schedule.add(cell)

