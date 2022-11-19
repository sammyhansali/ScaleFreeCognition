from mesa import Agent
import random
from enum import IntEnum
import numpy as np
import sys
import multicellularity.schedule


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
    energy = None
    energyt1=None
    state=None
    statet1=None
    collective_size=None
    moore = True   
    molecules = None
    goal=None
    cell_gain_from_good_state = None
    stress = None
    stresst1=None
    decision_state0 = None
    decision_state1= None
    decision_state2 = None
    GJ_opening_molecs = None
    GJ_opening_stress = None


    def __init__(self, net, depth, unique_id, pos, model, moore, molecules, energy, energyt1, cell_gain_from_good_state,  goal, GJ_opening_molecs, GJ_opening_stress, stress, stresst1, decision_state0, decision_state1, decision_state2, state, statet1, state_tissue):
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
        self.cell_gain_from_good_state = cell_gain_from_good_state
        # Channel Openings
        self.GJ_opening_molecs = GJ_opening_molecs # single variable for now, not sure if should be a list. One ele for each molec type?
        self.GJ_opening_stress = GJ_opening_stress
        # Inputs
        self.energy = energy
        self.energyt1 = energyt1
        self.state = state 
        self.statet1 = state 
        self.stress = stress
        self.stresst1 = stresst1
        self.state_tissue = state_tissue
        self.decision_state0 = decision_state0
        self.decision_state1 = decision_state1
        self.decision_state2 = decision_state2
        
    def net_inputs(self, finite_reservoir):
        inputs = []
        
        # Determining what you want as inputs
        if "finite_reservoir" in self.model.ANN_inputs:
            inputs.append(finite_reservoir)
        if "molecules" in self.model.ANN_inputs:
            inputs.append(list(self.molecules))
            inputs = sum(inputs, [])
        if "energy" in self.model.ANN_inputs:
            inputs.append(self.energy)
        if "energyt1" in self.model.ANN_inputs:
            inputs.append(self.energyt1)
        if "stress" in self.model.ANN_inputs:
            inputs.append(self.stress)
        if "stresst1" in self.model.ANN_inputs:
            inputs.append(self.stresst1)
        if "state" in self.model.ANN_inputs:
            inputs.append(self.state)
        if "statet1" in self.model.ANN_inputs:
            inputs.append(self.statet1) 
        # N = 1
        if "local_geometrical_frustration" in self.model.ANN_inputs:
            inputs.append(self.local_geometrical_frustration()) 
        if "local_state" in self.model.ANN_inputs:
            inputs.append(self.local_state()) 
        # N = 2 (come back later. Have to make sure it is bio relevant)
        # Raw positional data
        if "pos_x" in self.model.ANN_inputs:
            inputs.append(self.pos[0])
            # inputs.append(self.pos[0]/8)
        if "pos_y" in self.model.ANN_inputs:
            inputs.append(self.pos[1])
            # inputs.append(self.pos[1]/8)
        if "fitness_score" in self.model.ANN_inputs:
            inputs.append(self.model.schedule.fitness()/100)
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
    
    def update_state(self):
        if self.molecules[0] >=  10 :
            if self.state!=1:
                self.statet1 = self.state
                self.state = 1
            else: self.state = 1
           
        elif  self.molecules[0] < 10 and self.molecules[0] >= 5:
            if self.state!=3:
                self.statet1 = self.state
                self.state = 3
            else: self.state = 3

        elif self.molecules[0] >= 0 and self.molecules[0] < 5:
            if self.state!=2:
                self.statet1 = self.state
                self.state = 2    
            else: self.state = 2    
            
    def update_stress(self):
        if self.stress > 100:
            self.stress = 100
        if self.stress < 0:
            self.stress=0           
            
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
                    cell_in_contact.stress+=outputs['stress_to_send']*((self.GJ_opening_stress * cell_in_contact.GJ_opening_stress))   
                    cell_in_contact.stress-=outputs['anxio_to_send']*((self.GJ_opening_stress * cell_in_contact.GJ_opening_stress))   
                cell_in_contact.update_stress()

        if self.GJ_opening_stress>0:
             self.stress+=outputs['stress_to_send'] 
             self.stress-=outputs['anxio_to_send'] 

        # for i in range(len(self.molecules)):
        # if self.molecules[0] < 0:
        #     self.molecules[0] = 0

        self.update_state()   
        self.update_stress() # added in    


        
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
        
                
    def local_geometrical_frustration (self):
        geometrical_frustration = 0
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                 cell_in_contact = self.model.grid.get_cell_list_contents([neighbour])[0]
                 if self.state != cell_in_contact.state:
                     geometrical_frustration += 1
        return geometrical_frustration / (len(neighbours))

    # reward by concentric, or by N=1 nearest neighbours
    def local_state(self):
        local_state = 0
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                cell = self.model.grid.get_cell_list_contents([neighbour])[0]
                if cell.state == cell.goal:
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

    # def cell_division(self, probability):
    #     # if there is space to divide, use probability to decide whether to or not
    #     dead=[]
    #     neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
    #     for neighbour in neighbours:
    #         if self.model.grid.is_cell_empty(neighbour)==True:
    #             dead.append(neighbour)
    #     if len(dead) > 0:
    #         winner = random.choice(dead)
    #         # if probability == 1 and self.molecules[0] > 1:
    #         # print(probability) #testing line
    #         if probability == 1:
    #             # DIVIDE
    #             x = winner[0]
    #             y = winner[1]
    #             # Updating initial cell
    #             # self.molecules[0] /= 2
    #             self.energy /= 2
    #             self.stress /= 2
    #             # self.update_state()
    #             self.update_stress()
    #             # Creating new cell with half
    #             cell = Cell(self.net, self.depth, self.model.next_id(), (x,y), self.model,  True, 
    #                             self.molecules, self.energy, 0, self.cell_gain_from_good_state,  self.model.goal[y][x], 
    #                             self.GJ_opening_molecs, self.GJ_opening_stress, self.stress, 0, self.decision_state0, self.decision_state1, 
    #                             self.decision_state2, self.state, 0, self.state_tissue)
    #             self.birth(cell, x, y)
    def cell_division(self, output_prob, depleted_reservoir):
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        n = len(neighbours)
        # Get discrete numbers 0, 1, 2,..., n
        index = round( output_prob * n ) 

        # Don't divide if index == n
        if index == n:
            return
        # If desired position is empty, divide. Else can't divide.
        if self.model.grid.is_cell_empty(neighbours[index])==True:
            self.divide(neighbours[index])
            return depleted_reservoir-1

    def divide(self, pos):
        x = pos[0]
        y = pos[1]

        # Updating initial cell
        # self.molecules[0] /= 2
        # self.update_state()
        self.energy /= 2
        self.stress /= 2
        self.update_stress()
        cell = Cell(self.net, self.depth, self.model.next_id(), (x,y), self.model,  True, 
                        self.molecules, self.energy, 0, self.cell_gain_from_good_state,  self.model.goal[y][x], 
                        self.GJ_opening_molecs, self.GJ_opening_stress, self.stress, 0, self.decision_state0, self.decision_state1, 
                        self.decision_state2, self.state, 0, self.state_tissue)
        self.birth(cell, x, y)
        

    def step(self, depleted_reservoir, full_reservoir):
        """
        A model step. 
        """
        
        self.energyt1 = self.energy
        self.stresst1 = self.stress
        # Added in
        self.statet1 = self.state
                
        inputs = self.net_inputs()
        # Give appreciation to how much finite E is left!
        inputs.append(depleted_reservoir/full_reservoir) 
        outputs = self.prune_outputs(self.net_outputs(inputs))
        self.send_ions_and_stress(outputs)

        # self.energy -= 0.8
            
        if self.energy <= 0:
            self.die()
            return

        if "cell_division" in output:
            self.cell_division(output["cell_division"], depleted_reservoir)
        
        if "apoptosis" in output:
            self.apoptosis(output["apoptosis"])

        return depleted_reservoir
            
    def die(self):
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)

    def birth(self, cell, x, y): 
        self.model.grid.place_agent(cell, (x, y))
        self.model.schedule.add(cell)

        
