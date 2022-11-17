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
        
    def net_input(self):
           
        new_input = []
        
        # Determining what you want as inputs
        if "molecules" in self.model.ANN_inputs:
            new_input.append(list(self.molecules))
            new_input = sum(new_input, [])
        if "energy" in self.model.ANN_inputs:
            new_input.append(self.energy)
        if "energyt1" in self.model.ANN_inputs:
            new_input.append(self.energyt1)
        if "stress" in self.model.ANN_inputs:
            new_input.append(self.stress)
        if "stresst1" in self.model.ANN_inputs:
            new_input.append(self.stresst1)
        if "state" in self.model.ANN_inputs:
            new_input.append(self.state)
        if "statet1" in self.model.ANN_inputs:
            new_input.append(self.statet1) 
        # N = 1
        if "local_geometrical_frustration" in self.model.ANN_inputs:
            new_input.append(self.local_geometrical_frustration()) 
        if "local_state" in self.model.ANN_inputs:
            new_input.append(self.local_state()) 
        # N = 2

        # # Optional
        # if "collective_size" in self.model.ANN_inputs:
        #     new_input.append(self.collective_size)
        # if "french_flag" in self.model.ANN_inputs:
        #     new_input.append(np.abs(100 - (self.collective_size/(self.model.nb_goal_cells/3)*100)))
        
        # Raw positional data
        if "pos_x" in self.model.ANN_inputs:
            new_input.append(self.pos[0])
            # new_input.append(self.pos[0]/8)
        if "pos_y" in self.model.ANN_inputs:
            new_input.append(self.pos[1])
            # new_input.append(self.pos[1]/8)

        # Bias
        if "bias" in self.model.ANN_inputs:
            new_input.append(0.5)
        
        
        return(new_input)
    
    def net_output(self,new_input):
        
        #OUTPUT network    
        self.net.Flush()
        self.net.Input(new_input)
        [self.net.Activate() for _ in range(self.depth)]
        raw_output = list(self.net.Output())
        # Might want to try 4 different cell_division outputs
        # print(len(self.model.ANN_outputs)) #7
        # print(len(raw_output)) #5
        output = {k: v for k,v in zip(self.model.ANN_outputs, raw_output)}
        # print("cell_division" in output)
        
        return output
    
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
            
    def send_ions_and_stress(self, output):
 
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
                if self.molecules[i] >= output[key] * GJ_open_percentage:
                    cell_in_contact.molecules[i] += output[key] * GJ_open_percentage   
                    self.molecules[i] -= output[key] * GJ_open_percentage   
                else:
                    cell_in_contact.molecules[i] += self.molecules[i]
                    self.molecules[i] = 0
                cell_in_contact.update_state()  

                # stress to send
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
        self.update_stress() # added in    


        
    def communication(self):
        """Find cell neighbours and pass information and/or energy. It 
           represents basic features of gap junctions."""
                    
        new_input = self.net_input()
        output  = self.net_output(new_input)
        
        # supports multiple molecules sent out in future
        # for i in range(self.molecules):
        #     key = f"m{i+1}_to_send"
        key ="m0_to_send"
        if output[key] < 0:
            output[key] = 0
        if output["stress_to_send"] < 0:
            output["stress_to_send"] = 0
        if output["anxio_to_send"] < 0:
            output["anxio_to_send"] = 0
        
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

        # make apoptosis variable binary
        if "apoptosis" in output:
            if output["apoptosis"] < 0:
                output["apoptosis"] = 0
            elif output["apoptosis"] > 1:
                output["apoptosis"] = 1   
            output["apoptosis"] = round(output["apoptosis"]) # Want binary

        # make cell div variable binary
        if "cell_division" in output:
            if output["cell_division"] < 0:
                output["cell_division"] = 0
            elif output["cell_division"] > 1:
                output["cell_division"] = 1

        # send ions and stress
        self.send_ions_and_stress(output)
        return output
        
                
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

        # for y in range(self.model.height):
        #     for x in range(int(self.model.width)):
        #         local_state=0
        #         # stress=0
        #         n=1
        #         if len(self.model.grid.get_cell_list_contents([(x,y)]))>0:
        #             neighbours = self.model.grid.get_neighborhood(pos=(x,y), moore=True, include_center=True)
        #             n=len(neighbours)
        #             for neighbour in neighbours:
                        
        #                 if len(self.model.grid.get_cell_list_contents([neighbour]))>0:
        #                     cell = self.model.grid.get_cell_list_contents([(x,y)])[0]
        #                     if cell.state == cell.goal:
        #                         local_state+=1
        #                     # else:
        #                     #     stress+=1
        #                 else:
        #                     if self.model.goal[y][x] == 0:
        #                         local_state+=1
        #                     # else:
        #                     #     stress+=1
        #         local_state /= n
        #         # stress_mat[x,y] = stress / n
        # return local_state
        # return reward_mat, stress_mat
                     
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
    #             # self.energy /= 2
    #             self.stress /= 2
    #             # self.update_state()
    #             self.update_stress()
    #             # Creating new cell with half
    #             cell = Cell(self.net, self.depth, self.model.next_id(), (x,y), self.model,  True, 
    #                             self.molecules, 70, 0, self.cell_gain_from_good_state,  self.model.goal[y][x], 
    #                             self.GJ_opening_molecs, self.GJ_opening_stress, self.stress, 0, self.decision_state0, self.decision_state1, 
    #                             self.decision_state2, self.state, 0, self.state_tissue)
    #             self.birth(cell, x, y)
    def cell_division(self, output_prob):
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

    def divide(self, pos):
        x = pos[0]
        y = pos[1]

        # Updating initial cell
        # self.molecules[0] /= 2
        # self.update_state()
        # self.energy /= 2
        self.stress /= 2
        self.update_stress()
        cell = Cell(self.net, self.depth, self.model.next_id(), (x,y), self.model,  True, 
                        self.molecules, 70, 0, self.cell_gain_from_good_state,  self.model.goal[y][x], 
                        self.GJ_opening_molecs, self.GJ_opening_stress, self.stress, 0, self.decision_state0, self.decision_state1, 
                        self.decision_state2, self.state, 0, self.state_tissue)
        self.birth(cell, x, y)
        

    def step(self):
        """
        A model step. 
        """
        
        self.energyt1 = self.energy
        self.stresst1 = self.stress
        p = list(self.pos)
        # self.tissue = tissue_matrix[p[0], p[1]]
        # self.collective_size = tissue_matrix[p[0], p[1]]
        output = self.communication()
        # print(output.keys())


        # reward = reward_mat[self.pos]
        # stress = stress_mat[self.pos]
        # self.energy += reward - 0.8
        self.energy -= 0.8
            
        # if self.stress > 100:
        #     self.stress = 100
        # if self.stress < 0:
        #     self.stress=0
            
        if self.energy <= 0:
            self.die()
            return

        if "cell_division" in output:
            self.cell_division(output["cell_division"])
        
        if "apoptosis" in output:
            self.apoptosis(output["apoptosis"])
            
    def die(self):
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)

    def birth(self, cell, x, y): 
        self.model.grid.place_agent(cell, (x, y))
        self.model.schedule.add(cell)

        
