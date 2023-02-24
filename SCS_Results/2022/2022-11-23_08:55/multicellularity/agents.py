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
    # energy = None
    # energyt1=None
    # state=None
    # statet1=None
    moore = True   
    molecules = None
    goal=None
    # stress = None
    # stresst1=None
    GJ_opening_molecs = None
    GJ_opening_stress = None
    energy = []
    state = []
    stress = []
    # How about an energy list? energy[0] is cells energy at t=x. energy[1] is energy at t=x-1. Etc...
    # Similar idea with stress and state and molecules
    # And of course with tissue state


    # def __init__(self, net, depth, unique_id, pos, model, moore, molecules, energy, energyt1,  goal, GJ_opening_molecs, GJ_opening_stress, stress, stresst1, state, statet1):
    def __init__(self, net, depth, unique_id, pos, model, moore, molecules, goal, GJ_opening_molecs, GJ_opening_stress, energy, stress, state):
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
        # self.energy = energy
        # self.energyt1 = energyt1
        # self.state = state 
        # self.statet1 = state 
        # self.stress = stress
        # self.stresst1 = stresst1
        self.energy = energy
        self.state = state
        self.stress = stress
        
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
        if "finite_reservoir" in self.model.ANN_inputs:
            inputs.append(self.model.depleted_reservoir/self.model.full_reservoir) 
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
    
    def update_history(self, var, update):
        # add 'update' to index zero and shift everything down
        var = [update] + var[:-1]

    def update_state(self):
        if self.molecules[0] >=  10 :
            self.update_history(self.state, 1)
            # if doesn't work: self.state = [1] + self.state[:-1]
           
        elif  self.molecules[0] < 10 and self.molecules[0] >= 5:
            self.update_history(self.state, 3)

        elif self.molecules[0] < 5 and self.molecules[0] >= 0:
            self.update_history(self.state, 2)
            
    def prune_stress(self, stress):
        if stress > 100:
            stress = 100
        if stress < 0:
            stress=0     
        return
            
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
                    self.update_history( cell_in_contact.stress, 
                                    self.prune_stress(new_stress_amount)
                                    )
                    # cell_in_contact.stress += outputs['stress_to_send']*((self.GJ_opening_stress * cell_in_contact.GJ_opening_stress))   
                    # cell_in_contact.stress -= outputs['anxio_to_send']*((self.GJ_opening_stress * cell_in_contact.GJ_opening_stress))   

        if self.GJ_opening_stress>0:
            new_stress_amount = self.stress[0] + outputs['stress_to_send'] - outputs['anxio_to_send'] 
            self.update_history( self.stress, 
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
        # neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
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
    def cell_division(self, outputs):
        cd_prob = outputs["cell_division"]
        neighbours = self.model.grid.get_neighborhood(self.pos, self.moore, False)
        n = len(neighbours)
        # Get discrete numbers 0, 1, 2,..., n
        index = round( cd_prob * n ) 

        # Don't divide if index == n
        if index != n:
            # If desired position is empty, divide. Else can't divide.
            if self.model.grid.is_cell_empty(neighbours[index])==True:
                subtract=False
                if outputs["use_finite_reservoir"]==1:
                    subtract=True
                self.divide(neighbours[index], subtract)
                

    def divide(self, pos, subtract):
        x = pos[0]
        y = pos[1]

        # Updating initial cell
        # self.energy /= 2
        # self.stress /= 2
        # self.prune_stress()
        history_length=10
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
                        # energy = self.model.energy, 
                        # energyt1 = self.model.energy,  
                        energy = [0]*history_length,
                        # energy.append(self.model.energy)
                        # stress = self.stress, 
                        # stresst1 = 0, 
                        stress = [0]*history_length, 
                        # stress.append(self.stress)
                        # state = self.state, 
                        # statet1 = 0
                        state = [0]*history_length,
                        # state.append(self.state)
                    )
        cell.energy[0] = self.model.energy
        cell.stress[0] = self.stress
        cell.state[0] = self.state
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

        if self.energy[0] <= 0:
            self.die()
            return
        
        self.update_history(self.energy, self.energy[0])

        if "apoptosis" in outputs:
            self.apoptosis(outputs["apoptosis"])
            
        if "cell_division" in outputs:
            # self.cell_division(outputs["cell_division"])
            self.cell_division(outputs)
        
        
        

    def die(self):
        # self.model.depleted_reservoir += 1    # Testing if increasing reservoir after death makes sense
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)

    def birth(self, cell, x, y, subtract): 
        # if subtract:
            # self.model.depleted_reservoir -= 1
        self.model.depleted_reservoir -= 1
        self.model.grid.place_agent(cell, (x, y))
        self.model.schedule.add(cell)

        
