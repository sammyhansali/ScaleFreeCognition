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
        #
        self.molecules = molecules # list, one for each molec type
        self.goal = goal
        self.cell_gain_from_good_state = cell_gain_from_good_state
        # Channel Openings
        self.GJ_opening_molecs = GJ_opening_molecs # single variable for now, not sure if should be a list
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
        
        
    def net_input(self, perc_blue, perc_red, perc_white, fitness_ff, tissue_matrix):


        # #INPUT network      
        new_input = []
        new_input.append(list(self.molecules))
        new_input = sum(new_input, [])
        new_input.append(self.energy)
        new_input.append(self.energyt1)
        new_input.append(self.stress)
        new_input.append(self.stresst1)
        new_input.append(self.state)
        new_input.append(self.statet1)  
        new_input.append(self.local_geometrical_frustration())  
        pos = list(self.pos)

        new_input.append(tissue_matrix[pos[0], pos[1]])   # how big the collective is
        new_input.append(np.abs(100 - (tissue_matrix[pos[0], pos[1]]/(self.model.height*self.model.width/3)*100)))


        new_input.append(0.5) # the bias 

        return(new_input)
    
    def net_output(self,new_input):
        
        #OUTPUT network    
        self.net.Flush()
        #print(self.net_activations)
        self.net.Input(new_input)
        [self.net.Activate() for _ in range(self.depth)]
        raw_output = list(self.net.Output())  
        output_tags = ["m0_to_send", "GJ_opening_molecs", "stress_to_send", "GJ_opening_stress", "anxio_to_send", "apoptosis", "cell_division"] 
        # Might want to try 4 different cell_division outputs
        output = {k: v for k,v in zip(raw_output, output_tags)}
        
        return output
    
    def update_state(self):
        
        if self.molecules[0] >=  10 :
            if self.state!=1:
                #self.energy -=0.5
                self.statet1 = self.state
                self.state = 1
            else: self.state = 1
           # else:
           #     self.energy -=0.25
           
        elif  self.molecules[0] < 10 and self.molecules[0] >= 5:
            if self.state!=3:
                #self.energy -=0.5
                self.statet1 = self.state
                self.state = 3
            else: self.state = 3

        elif self.molecules[0] >= 0 and self.molecules[0] < 5:
            if self.state!=2:
                #self.energy -=0.5
                self.statet1 = self.state
                self.state = 2    
            else: self.state = 2    
           # else:
           #     self.energy -=0.25
            
    def update_stress(self):
    
        if self.stress > 100:
            self.stress = 100
        if self.stress < 0:
            self.stress=0           
            
    def send_ions_and_stress(self, output):
 
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        
        # Molecule 1 distribution
        # for i in range(self.molecules):
        #     key = f"m{i}_to_send"
        i=0                 # placeholder, later will put loop like above
        key="m0_to_send"    # placeholder, later will put loop like above
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                cell_in_contact = self.model.grid[neighbour][0]
                
                GJ_open_percentage = min(self.GJ_opening_molecs, cell_in_contact.GJ_opening_molecs)
                if self.molecules[i] >= output[key] * GJ_open_percentage:
                    cell_in_contact.molecules[i] += output[key] * GJ_open_percentage   
                    self.molecules[i] -= output[key] * GJ_open_percentage   
                else:
                    cell_in_contact.molecules[i] += self.molecules[i]
                    self.molecules[i] = 0
                cell_in_contact.update_stress()         
        
        # Stress + anxiolytic distribution
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                 cell_in_contact = self.model.grid[neighbour][0]
                
                 if self.GJ_opening_stress>0:
                     cell_in_contact.stress+=output['stress_to_send']*((self.GJ_opening_stress * cell_in_contact.GJ_opening_stress))   
                     cell_in_contact.stress-=output['anxio_to_send']*((self.GJ_opening_stress * cell_in_contact.GJ_opening_stress))   
                 cell_in_contact.update_stress()
        if self.GJ_opening_stress>0:
             self.stress+=output['stress_to_send'] 
             self.stress-=output['anxio_to_send'] 

        # for i in range(len(self.molecules)):
        # if self.molecules[0] < 0:
        #     self.molecules[0] = 0

        self.update_state()       


        
    def communication(self, perc_blue, perc_red, perc_white, fitness_ff, tissue_matrix):
        """Find cell neighbours and pass information and/or energy. It 
           represents basic features of gap junctions."""
                    
        new_input = self.net_input(perc_blue, perc_red, perc_white, fitness_ff, tissue_matrix)
        output  = self.net_output(new_input)
        
        # supports multiple molecules sent out in future
        # for i in range(self.molecules):
        #     key = f"m{i+1}_to_send"
        key ="m0_to_send"
        if output[key] < 0:
            output[key] = 0
        if output["stress_to_send"] < 0:
            output["stress_to_send"] = 0
        
        # molecules1 GJ opening correction
        self.GJ_opening_molecs = output['GJ_opening_molecs']
        if self.GJ_opening_molecs < 0:
            self.GJ_opening_molecs = 0
        elif self.GJ_opening_molecs > 1:
            self.GJ_opening_molecs = 1

        # stress GJ opening correction
        self.GJ_opening_stress = output['GJ_opening_stress']
        if self.GJ_opening_stress < 0:
            self.GJ_opening_stress = 0
        elif self.GJ_opening_stress > 1:
            self.GJ_opening_stress = 1           

        # send ions and stress
        self.send_ions_and_stress(output)
        return output
        
                
    def local_geometrical_frustration (self):
        geometrical_frustration = 0
        dead = 0
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                 cell_in_contact = self.model.grid.get_cell_list_contents([neighbour])[0]
                 if self.state != cell_in_contact.state:
                     geometrical_frustration += 1
            else:
                dead += 1

        return geometrical_frustration / (len(neighbours))
                     

                    
    def step(self, reward_mat, stress_mat, perc_blue, perc_red, perc_white, fitness_ff, tissue_matrix):
        """
        A model step. 
        """
        
        self.energyt1 = self.energy
        self.stresst1 = self.stress
        output = self.communication(perc_blue, perc_red, perc_white, fitness_ff, tissue_matrix)


        reward = reward_mat[self.pos]
        stress = stress_mat[self.pos]

        self.energy += reward - 0.8

            
        if self.stress > 100:
            self.stress = 100
        if self.stress < 0:
            self.stress=0
            

        
        # Death
        if self.energy <= 0:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)
            
            
        
