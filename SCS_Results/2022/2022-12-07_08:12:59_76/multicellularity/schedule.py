from collections import defaultdict
from mesa.time import RandomActivation
import numpy as np
from spatialentropy import leibovici_entropy, altieri_entropy
import random 
from scipy.ndimage.measurements import label


class RandomActivationByBreed(RandomActivation):
    """
    A scheduler which activates each type of agent once per step, in random
    order, with the order reshuffled every step.

    This is equivalent to the NetLogo 'ask breed...' and is generally the
    default behavior for an ABM.

    Assumes that all agents have a step() method.
    """

    def __init__(self, model):
        super().__init__(model)
        self.agents_by_breed = defaultdict(dict)

    def add(self, agent):
        """
        Add an Agent object to the schedule

        Args:
            agent: An Agent to be added to the schedule.
        """

        self._agents[agent.unique_id] = agent
        agent_class = type(agent)
        self.agents_by_breed[agent_class][agent.unique_id] = agent

    def remove(self, agent):
        """
        Remove all instances of a given agent from the schedule.
        """

        del self._agents[agent.unique_id]

        agent_class = type(agent)
        del self.agents_by_breed[agent_class][agent.unique_id]

    # Try this soon. After figuring out input modularity.
    def cell_contents_at(j, i):
        return self.model.grid.get_cell_list_contents([(j,i)])


    def step(self, Cell):
        """
        Executes the step of each agent breed, one at a time, in random order.

        Args:
            by_breed: If True, run all agents of a single breed before running
                        the next one.
        """
        agent_keys = list(self._agents.keys())
        self.model.random.shuffle(agent_keys)
        for key in agent_keys:
            self._agents[key].step()
        self.steps += 1
        self.time += 1


    def get_breed_count(self, breed_class):
        """
        Returns the current number of agents of certain breed in the queue.
        """
        return len(self.agents_by_breed[breed_class].values())
        #Optimize by removing ".values()"
    
    
    def get_global_stress(self, Cell):
        
        stress = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())
        for agent_key in agent_keys:
            if self.agents_by_breed[Cell][agent_key].cell_type[0] != self.agents_by_breed[Cell][agent_key].goal:
                stress+=1
        return stress

    def get_internal_stress(self, Cell):
        
        stress = 0
        for i in range(self.model.height):
            for j in range(self.model.width):
                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell= self.model.grid.get_cell_list_contents([(j,i)])[0] 
                    stress+=cell.stress[0]
        return stress
        
    def get_open_cells(self, Cell):
        
        open_cells = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())
        for agent_key in agent_keys:
            if self.agents_by_breed[Cell][agent_key].GJ_opening_ions == 1:
                open_cells+=1
        return open_cells       
    
    def get_spatial_entropy(self, Cell):
        agent_keys = list(self.agents_by_breed[Cell].keys())
        points = []
        cell_types = []
        for i in range(self.model.height):
            for j in range(self.model.width):
                for agent_key in agent_keys:
                    if self.agents_by_breed[Cell][agent_key].pos[0] == j and self.agents_by_breed[Cell][agent_key].pos[0] == i:
                        points.append([j,i])
                        cell_types.append( self.agents_by_breed[Cell][agent_key].cell_type[0])
        points = np.array(points)
        cell_types = np.array(cell_types)
        e = altieri_entropy(points, cell_types)
        return e.entropy
             
    def local_fitness(self, cell):
        local_fit = 0
        # Put true so score is lower if the center cell is not in correct cell_type.
        neighbours = self.model.grid.get_neighborhood(cell.pos, cell.moore, True)
        for neighbour in neighbours:
            if self.model.grid.is_cell_empty(neighbour)==False: # not dead
                cell_in_contact = self.model.grid.get_cell_list_contents([neighbour])[0]
                if cell_in_contact.cell_type[0] == cell_in_contact.goal:
                    local_fit += 1
            else:
                # Need to get x and y coords of the dead neighbour
                x=neighbour[0]
                y=neighbour[1]
                if self.model.goal[y][x] == 0:
                    local_fit += 1
        return local_fit / (len(neighbours)) * 100

    def global_fitness(self):
        self.fitness_score=0
        for i in range(self.model.height):
            for j in range(int(self.model.width)):
                # alive cell on spot
                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                    if cell.cell_type[0] == cell.goal:
                        self.fitness_score+=1
                # nothing on spot
                else:
                    if self.model.goal[i][j] == 0:
                        self.fitness_score+=1
        
        self.fitness_score = self.fitness_score/(self.model.height*self.model.width)*100
        return self.fitness_score

    # def global_fitness_2(self):
    #     self.fitness_score=0
    #     for i in range(self.model.height):
    #         for j in range(int(self.model.width)):
    #             # alive cell on spot
    #             if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                 cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                 if cell.goal == 0:
    #                     self.fitness_score-=1
    #                 elif cell.state[0] == cell.goal:
    #                     self.fitness_score+=1
    #             # nothing on spot
    #             # else:
    #             #     if self.model.goal[i][j] == 0:
    #             #         self.fitness_score+=1
        
    #     self.fitness_score = self.fitness_score/(self.model.nb_goal_cells)*100
    #     return self.fitness_score

    # def global_fitness_3(self):
    #     correct=0
    #     tot=0
    #     for i in range(self.model.height):
    #         for j in range(int(self.model.width)):
    #             # alive cell on spot
    #             if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                 tot += 1
    #                 cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                 if cell.state[0] == cell.goal:
    #                     correct+=1
        
    #     patterning = 50*(correct/self.model.nb_goal_cells) 
        # size = 50 - (50 * abs(tot/self.model.nb_goal_cells - 1))
        # self.fitness_score = patterning + size
        # return self.fitness_score
    
    # def pos_occupied_at(self, x, y):
    #     return len(self.model.grid.get_cell_list_contents([(x,y)]))>0
    
    
    def general_geometrical_frustration(self, Cell):
        general_frustration=0
        for i in range(self.model.height):
             for j in range(int(self.model.width)):
                 local_frustration =0

                 if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                     cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                     neighbours = self.model.grid.get_neighborhood((j,i), True, True)
                     nb_neighbour=0
                     for neighbour in neighbours:
                         if len(self.model.grid.get_cell_list_contents([neighbour]))>0:
                             nb_neighbour+=1
                             cell_neighbour= self.model.grid.get_cell_list_contents([(j,i)])[0]
                             if cell.cell_type[0] == cell_neighbour.cell_type[0]:
                                 local_frustration+=1
                                 break
                 general_frustration+= local_frustration
                         
        
        return general_frustration

    
    
    def generate_cells(self, Cell, start):
        # Use start mat to generate starting cells
        for y in range(self.model.height):
            for x in range(self.model.width):

                # Setting up communication network
                molecules = {}
                for n in range(self.model.nb_output_molecules):
                    molecules[n] = [0]*self.model.history_length
                    # Arbitrary amount of signaling molecules to begin with
                    molecules[n][0] = 5


                cell_type = start[y][x]
                resting_pot = 0
                if cell_type ==     1:      # Skin
                    resting_pot = -60
                elif cell_type ==   2:      # Epithelial
                    resting_pot = -50
                elif cell_type ==   3:      # Retinal
                    resting_pot = -40
                # amount = 0
                # if state_cell == 0:
                #     continue
                # elif state_cell == 3:
                #     amount = 9 # So if each cell has 1 type of molec, total will be 9. If 2, then 18. If 3, then 27, etc.
                # elif state_cell == 2:
                #     amount = 6
                # elif state_cell == 1:
                #     amount = 3
                # for x in range(len(molecules)):
                #     molecules[x][0] = amount

                cell = Cell(    net = self.model.net, 
                                depth = self.model.depth, 
                                unique_id = self.model.next_id(), 
                                pos = (x, y), 
                                model = self.model,  
                                moore = True, 
                                goal = self.model.goal[y][x], 
                                GJ_opening_ions=0, 
                                GJ_opening_stress=0, 
                                # Historical data
                                energy = [0]*self.model.history_length,
                                stress = [0]*self.model.history_length, 
                                local_fitness = [0]*self.model.history_length,
                                global_fitness = [0]*self.model.history_length,
                                direction = [0]*self.model.history_length,
                                cell_type = [0]*self.model.history_length,
                                potential = [0]*self.model.history_length,
                                molecules = molecules, 
                            )
                cell.energy[0] = self.model.energy
                cell.global_fitness[0] = self.global_fitness()/100
                cell.direction[0] = random.choice(range(1,9))
                cell.cell_type[0] = cell_type
                cell.potential[0] = resting_pot
                self.model.grid.place_agent(cell, (x, y))
                self.model.schedule.add(cell)
    
                     
                
                     
