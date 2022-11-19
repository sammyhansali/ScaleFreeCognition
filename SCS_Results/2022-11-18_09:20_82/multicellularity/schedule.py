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

    # def step(self, Cell, reward_mat, stress_mat, perc_blue, perc_red, perc_white, fitness_score, tissue_matrix,  by_breed=True):
    # def step(self, Cell, reward_mat, stress_mat, fitness_score, tissue_matrix,  by_breed=True):
    # def step(self, Cell, by_breed=True):
    def step(self, Cell, by_breed=True):
        """
        Executes the step of each agent breed, one at a time, in random order.

        Args:
            by_breed: If True, run all agents of a single breed before running
                      the next one.
        """
        if by_breed:
            for agent_class in self.agents_by_breed:
                # self.step_breed(Cell, agent_class, reward_mat, stress_mat, perc_blue, perc_red, perc_white, fitness_score, tissue_matrix)
                self.step_breed(Cell, agent_class)
            self.steps += 1
            self.time += 1
        else:
            super().step()

    # def step_breed(self, Cell, breed, reward_mat, stress_mat, perc_blue, perc_red, perc_white, fitness_score, tissue_matrix ):
    # def step_breed(self, Cell, breed, reward_mat, stress_mat, fitness_score, tissue_matrix ):
    def step_breed(self, Cell, breed):
        """
        Shuffle order and run all agents of a given breed.

        Args:
            breed: Class object of the breed to run.
        """

        agent_keys = list(self.agents_by_breed[breed].keys())
        self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            # self.agents_by_breed[breed][agent_key].step(reward_mat, stress_mat, perc_blue, perc_red, perc_white, fitness_score, tissue_matrix)
            # self.agents_by_breed[breed][agent_key].step(reward_mat, stress_mat, fitness_score, tissue_matrix)
            self.agents_by_breed[breed][agent_key].step()




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
            if self.agents_by_breed[Cell][agent_key].state != self.agents_by_breed[Cell][agent_key].goal:
                stress+=1
        return stress

    def get_internal_stress(self, Cell):
        
        stress = 0
        for i in range(self.model.height):
            for j in range(self.model.width):
                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell= self.model.grid.get_cell_list_contents([(j,i)])[0]                        
                    stress+=cell.stress
        return stress
        
    def get_open_cells(self, Cell):
        
        open_cells = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())
        for agent_key in agent_keys:
            if self.agents_by_breed[Cell][agent_key].GJ_opening_molecs == 1:
                open_cells+=1
        return open_cells       
    
    # def percentages_french_flag (self):

    #     blue=0
    #     white=0
    #     red=0

    #     for i in range(self.model.height):
    #         for j in range(int(self.model.width/3)):
    #            if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                if cell.state == 1:
    #                    blue+=1


    #     for i in range(self.model.height):
    #          for j in range(int(self.model.width/3), int(2*self.model.width/3)):
    #             if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                 cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                 if cell.state == 3:
    #                     white+=1

        
    #     for i in range(self.model.height):
    #          for j in range(int(2*self.model.width/3), self.model.width):
    #             if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                 cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                 if cell.state == 2:
    #                     red+=1
 
    #     perc_blue = (blue/(self.model.height*self.model.width/3))*100
    #     perc_white = (white/(self.model.height*self.model.width/3))*100
    #     perc_red = (red/(self.model.height*self.model.width)/3)*100
    #     # perc_blue = (blue/(self.model.nb_goal_cells/3))*100
    #     # perc_white = (white/(self.model.nb_goal_cells/3))*100
    #     # perc_red = (red/(self.model.nb_goal_cells)/3)*100
        
    #     return perc_blue, perc_red, perc_white     

    # Honestly have no clue what this does...
    # def adaptive_tissue(self):
        
    #     openedGJ_matrix = np.zeros ((self.model.height, self.model.width))        
    #     state_matrix = np.zeros ((self.model.height, self.model.width))
    #     stress_matrix = np.zeros ((self.model.height, self.model.width))
    #     tissue_matrix =  np.zeros ((self.model.height, self.model.width))
    #     energy_matrix =  np.zeros ((self.model.height, self.model.width))
    #     molecules_matrix =  np.zeros ((self.model.height, self.model.width))

    #     for i in range(self.model.height):
    #         for j in range(int(self.model.width)):
    #             if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                 cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #             #   if sum(cell.GJ_opening_molecs)>0:
    #                 if cell.GJ_opening_molecs > 0:
    #                     openedGJ_matrix[j,i]=1    
    #                 state_matrix[j,i] = cell.state_tissue
    #                 stress_matrix[j,i] = cell.stress
    #                 energy_matrix[j,i] = cell.energy
    #                 molecules_matrix[j,i] = cell.molecules[0]

    
    #     state_matrix1 = np.where((state_matrix!=1), 0, state_matrix)
    #     state_matrix2 = np.where((state_matrix!=2), 0, state_matrix)
    #     state_matrix3 = np.where((state_matrix!=3), 0, state_matrix)

    #     state_matrices = [state_matrix1, state_matrix2, state_matrix3]

    #     structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter


    #     for j in range(len(state_matrices)):
    #         labeled, ncomponents = label(np.array(state_matrices[j]), structure)
    #         indices = np.indices(state_matrices[j].shape).T[:,:,[1, 0]]
    #         if ncomponents > 0:
    #             for i in range (ncomponents):
    #                 positions = indices[labeled == i+1]
    #                 for k in range(len(positions)):
    #                     tissue_matrix[positions[k][0], positions[k][1] ] = len(positions)

       
    #     return tissue_matrix, state_matrix, stress_matrix, energy_matrix, molecules_matrix
            
                       
    # def adaptive_tissue1(self):
        
    #     openedGJ_matrix = np.zeros ((self.model.height, self.model.width))        
    #     state_matrix = np.zeros ((self.model.height, self.model.width))
    #     tissue_matrix =  np.zeros ((self.model.height, self.model.width))
    #     molecules_per_tissue0 = np.zeros ((self.model.height, self.model.width))
    #     molecules_per_tissue1 = np.zeros ((self.model.height, self.model.width))

    #     for i in range(self.model.height):
    #         for j in range(int(self.model.width)):
    #             if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                 cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                 # if sum(cell.GJ_opening_molecs)>0:
    #                 if cell.GJ_opening_molecs>0:
    #                     openedGJ_matrix[j,i]=1    
    #                     state_matrix[j,i] = cell.state

    
    #     state_matrix1 = np.where((state_matrix!=1), 0, state_matrix)
    #     state_matrix2 = np.where((state_matrix!=2), 0, state_matrix)
    #     state_matrix3 = np.where((state_matrix!=3), 0, state_matrix)

    #     state_matrices = [state_matrix1, state_matrix2, state_matrix3]

    #     structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter


    #     for j in range(len(state_matrices)):
    #         labeled, ncomponents = label(np.array(state_matrices[j]), structure)
    #         indices = np.indices(state_matrices[j].shape).T[:,:,[1, 0]]
    #         if ncomponents > 0:
    #             for i in range (ncomponents):
    #                 positions = indices[labeled == i+1]
    #                 molecules0 = 0
    #                 molecules1 = 0
    #                 for k in range(len(positions)):
    #                     tissue_matrix[positions[k][0], positions[k][1] ] = len(positions)
    #                     cell = self.model.grid.get_cell_list_contents([(positions[k][0], positions[k][1])])[0]
    #                     molecules0 += cell.molecules[0]
    #                     molecules1 += cell.molecules[1]
    #                 for k in range(len(positions)):
    #                     molecules_per_tissue0[positions[k][0], positions[k][1] ] = molecules0
    #                     molecules_per_tissue1[positions[k][0], positions[k][1] ] = molecules1

       
    #     return tissue_matrix, molecules_per_tissue0, molecules_per_tissue1
    
    def get_spatial_entropy(self, Cell):
        agent_keys = list(self.agents_by_breed[Cell].keys())
        points = []
        states = []
        for i in range(self.model.height):
            for j in range(self.model.width):
                for agent_key in agent_keys:
                    if self.agents_by_breed[Cell][agent_key].pos[0] == j and self.agents_by_breed[Cell][agent_key].pos[0] == i:
                        points.append([j,i])
                        states.append( self.agents_by_breed[Cell][agent_key].state)
        points = np.array(points)
        states = np.array(states)
        e = altieri_entropy(points, states)
        return e.entropy
             
        
    # def get_pca_goal(self, Cell):
    #     goal = []
    #     for i in range(self.model.height):
    #         for j in range(self.model.width):
    #             if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                 cell= self.model.grid.get_cell_list_contents([(j,i)])[0]                        
    #                 goal.append(cell.state)
    #             else:
    #                 goal.append(5)

    #     return goal
        
    # def get_dead_cells_and_regenerate(self, Cell, net, depth, unique_id, model, energy, energyt1 , cell_gain_from_good_state ,  molecules, goal, GJ_opening_molecs, GJ_opening_stress, stress, decision_state0, decision_state1, decision_state2, state, statet1, state_tissue):
                            
                            
    #     for i in range(self.model.height):
    #         for j in range(self.model.width):
    #             if len(self.model.grid.get_cell_list_contents([(j,i)]))==0: # dead cell
    #                 neighbours = self.model.grid.get_neighborhood((j,i), True, True)
    #                 for neighbour in neighbours:
    #                     if len(self.model.grid.get_cell_list_contents([neighbour]))>0: #living cell*
    #                         print("DONE")
    #                  #[random.random()*2 for i in range(2)] 

    #                         cell = Cell(net, depth, unique_id, (j,i), model,  True, 
    #                           molecules, energy, energyt1, cell_gain_from_good_state,  goal[i][j], 
    #                           GJ_opening_molecs,GJ_opening_stress, stress, decision_state0, decision_state1, 
    #                           decision_state2, state, statet1, state_tissue)
    #                         model.grid.place_agent(cell, (j, i))
    #                         model.schedule.add(cell)                            
    #                         break
      
    # def french_flag(self):
    def fitness(self):
        self.fitness_score=0
        for i in range(self.model.height):
            for j in range(int(self.model.width)):
                # alive cell on spot
                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                    if cell.state == cell.goal:
                        self.fitness_score+=1
                # nothing on spot
                else:
                    if self.model.goal[i][j] == 0:
                        self.fitness_score+=1
        
        self.fitness_score = self.fitness_score/(self.model.height*self.model.width)*100
        # self.fitness_score = self.fitness_score/(self.model.nb_goal_cells)*100
        
        return self.fitness_score
    
    # def get_global_state(self):
    #     global_state = np.zeros((self.model.height, self.model.width))
    #     for y in range(self.model.height):
    #         for x in range(int(self.model.width)):
    #             if len(self.model.grid.get_cell_list_contents([(x,y)]))>0:
    #                 cell = self.model.grid.get_cell_list_contents([(x,y)])[0]
    #                 global_state[x,y] = cell.state
    #             else:
    #                 global_state[x,y] = 0
    #     return global_state
    
    def pos_occupied_at(self, x, y):
        return len(self.model.grid.get_cell_list_contents([(x,y)]))>0
    
    # def geometrical_frustration(self, Cell):
    #     frustration = 0
    #     for i in range(self.model.height):
    #          for j in range(int(self.model.width/3)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  neighbours = self.model.grid.get_neighborhood((j,i), True, True)
    #                  for neighbour in neighbours:
    #                      if len(self.model.grid.get_cell_list_contents([neighbour]))>0:
    #                          cell_neighbour= self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                          if cell.state == cell_neighbour.state:
    #                              frustration = frustration
    #                          else:
    #                              frustration+=1
    #                              break
                         
                    
    #          for j in range(int(self.model.width/3+1), int(2*self.model.width/3)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  neighbours = self.model.grid.get_neighborhood((j,i), True, True)
    #                  for neighbour in neighbours:
    #                      if len(self.model.grid.get_cell_list_contents([neighbour]))>0:
    #                          cell_neighbour= self.model.grid.get_cell_list_contents([neighbour])[0]
    #                          if cell.state == cell_neighbour.state:
    #                              frustration = frustration
    #                          else:
    #                              frustration+=1
    #                              break
            
    #          for j in range(int(2*self.model.width/3 + 1), self.model.width):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  neighbours = self.model.grid.get_neighborhood((j,i), True, True)
    #                  for neighbour in neighbours:
    #                      if len(self.model.grid.get_cell_list_contents([neighbour]))>0:
    #                          cell_neighbour= self.model.grid.get_cell_list_contents([neighbour])[0]
    #                          if cell.state == cell_neighbour.state:
    #                              frustration = frustration
    #                          else:
    #                              frustration+=1
    #                              break

    #     return frustration
    
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
                             if cell.state == cell_neighbour.state:
                                 local_frustration+=1
                                 break
                 general_frustration+= local_frustration
                         
        
        return general_frustration

    # def get_cellular_stress(self, Cell):
        
    #     stress = 0
    #     agent_keys = list(self.agents_by_breed[Cell].keys())
    #     for agent_key in agent_keys:
    #             stress+=self.agents_by_breed[Cell][agent_key].stress
    #     return stress
        
    # def get_GJ(self):
        
    #     GJ_matrix0 = np.zeros ((self.model.height, self.model.width))     
    #     GJ_matrix1 = np.zeros ((self.model.height, self.model.width))        
    #     GJ_matrix2 = np.zeros ((self.model.height, self.model.width))        
    #     GJ_matrix3 = np.zeros ((self.model.height, self.model.width)) 
 
    #     for i in range(self.model.height):
    #         for j in range(int(self.model.width)):
    #             if len(self.model.grid.get_cell_list_contents([(j,i)]))>0: 
    #                 cell = self.model.grid.get_cell_list_contents([(j,i)])[0]    
    #                 GJ_matrix0[i][j] =  cell.GJ_opening_molecs
    #                 GJ_matrix1[i][j] =  cell.GJ_opening_molecs
    #                 GJ_matrix2[i][j] =  cell.GJ_opening_molecs
    #                 GJ_matrix3[i][j] =  cell.GJ_opening_molecs
    #     return GJ_matrix0, GJ_matrix1, GJ_matrix2, GJ_matrix3  
    
    # def scientific_pca(self, Cell):
    #     v1=0
    #     v2=0
    #     v3=0
    #     for i in range(self.model.height):
    #          for j in range(int(self.model.width/3)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state == cell.goal:
    #                      v1+=1

                    
    #          for j in range(int(self.model.width/3+1), int(2*self.model.width/3)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state == cell.goal:
    #                      v2+=1  
            

            
            
    #          for j in range(int(2*self.model.width/3 + 1), self.model.width):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state == cell.goal:
    #                      v3+=1
        
    #     return [v1, v2, v3]
    
    # def get_good_states(self, Cell):
    #     fitness=0
    #     for i in range(self.model.height):
    #          for j in range(int(self.model.width/3)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state == cell.goal:
    #                      fitness+=1

                    
    #          for j in range(int(self.model.width/3+1), int(2*self.model.width/3)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state == cell.goal:
    #                      fitness+=1  
            

            
            
    #          for j in range(int(2*self.model.width/3 + 1), self.model.width):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state == cell.goal:
    #                      fitness+=1
                         
                        

    #     return fitness

  
        

    
    # def get_proportion(self, Cell):
    #     blue=0
    #     white=0
    #     red=0
    #     for i in range(self.model.height):
    #          for j in range(int(self.model.width)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state == 1:
    #                      blue+=1
    #                  elif cell.state == 0:
    #                      white+=1
    #                  elif cell.state == 2:
    #                      red+=1
        
    #     perc_blue = (blue/(self.model.height*self.model.width))*100
    #     perc_white = (white/(self.model.height*self.model.width))*100
    #     perc_red = (red/(self.model.height*self.model.width))*100
        
    #     reward=100 - np.abs(33-perc_blue) - np.abs(33-perc_white) - np.abs(33-perc_red)
    #     reward = reward/100*5
    #     return reward
    
    
    
    # def reward_by_patches1(self,pos):
        
    #     width_cell = pos[0]
    #     reward=0
    #     stress=0
    #     not_dead=1
        
    #     if width_cell <= int(self.model.width/3):
    #         for i in range(self.model.height):
    #             for j in range(int(self.model.width/3)):
    #                  if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                      not_dead+=1
    #                      cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                      if cell.state == cell.goal:
    #                          reward+=1
    #                      else:
    #                          stress+=1

    #     elif width_cell >= int(self.model.width/3+1) and width_cell <= int(2*self.model.width/3):
    #         for i in range(self.model.height):
    #              for j in range(int(self.model.width/3+1), int(2*self.model.width/3)):
    #                  if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                      not_dead+=1
    #                      cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                      if cell.state == cell.goal:
    #                              reward+=1
    #                      else:
    #                          stress+=1
            
    #     elif width_cell >= int(2*self.model.width/3 + 1) and width_cell <= self.model.width:
    #         for i in range(self.model.height):
    #              for j in range(int(2*self.model.width/3 + 1), self.model.width):
    #                  if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                      not_dead+=1
    #                      cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                      if cell.state == cell.goal:
    #                              reward+=1
    #                      else:
    #                          stress+=1
                             
    #     return reward/not_dead, stress/not_dead

    # def update_state_tissue(self):
    #     for i in range(self.model.height):
    #          for j in range( self.model.width):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                 cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                 if cell.molecules[0] >= 10 :
    #                     cell.state_tissue = 1
            
    #                 elif cell.molecules[0] < 10 and cell.molecules[0] >= 5:
    #                     cell.state_tissue = 3    
                        
    #                 elif cell.molecules[0] >= 0 and cell.molecules[0] < 5:
    #                     cell.state_tissue = 2
                        
    # def update_state_tissue_costStateChange(self):
    #     for i in range(self.model.height):
    #          for j in range( self.model.width):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                 cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                 if cell.molecules[0] >= 10 :
    #                     if cell.state_tissue != 1:
    #                         cell.state_tissue = 1
    #                         cell.energy -=0.25
            
    #                 elif cell.molecules[0] < 10 and cell.molecules[0] >= 5:
    #                     if cell.state_tissue != 3:
    #                         cell.state_tissue = 3
    #                         cell.energy -=0.25
                        
    #                 elif cell.molecules[0] >= 0 and cell.molecules[0] < 5:
    #                     if cell.state_tissue != 2:
    #                         cell.state_tissue = 2
    #                         cell.energy -=0.25
                         
    # Problematic
    # def reward_by_patches(self):
        
    #     reward_mat=np.zeros((self.model.width, self.model.height))
    #     stress_mat=np.zeros((self.model.width, self.model.height))
        
    #     reward=0
    #     stress = 0
    #     not_dead=1
    #     stripe1 = []
    #     for i in range(self.model.height):
    #         for j in range(int(self.model.width/3)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  not_dead+=1
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state_tissue == cell.goal:
    #                      reward+=1
    #                      stripe1.append(cell.pos)
    #                  else:
    #                      stress+=1
    #     for i in range(self.model.height):
    #         for j in range(int(self.model.width/3)):
    #             reward_mat[j,i] = reward/not_dead
    #             stress_mat[j,i] = stress/not_dead


    #     reward=0
    #     stress = 0
    #     not_dead=1
    #     stripe2 = []
    #     for i in range(self.model.height):
    #          for j in range(int(self.model.width/3), int(2*self.model.width/3)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  not_dead+=1
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state_tissue == cell.goal:
    #                      reward+=1
    #                      stripe2.append(cell.pos)
    #                  else:
    #                      stress+=1
    #     for i in range(self.model.height):
    #          for j in range(int(self.model.width/3), int(2*self.model.width/3)):            
    #              reward_mat[j,i] = reward/not_dead
    #              stress_mat[j,i] = stress/not_dead


    #     reward=0
    #     stress = 0
    #     not_dead=1
    #     stripe3 = []
    #     for i in range(self.model.height):
    #          for j in range(int(2*self.model.width/3), self.model.width):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  not_dead+=1
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state_tissue == cell.goal:
    #                      reward+=1
    #                      stripe3.append(cell.pos)

    #                  else:
    #                      stress+=1
    #     for i in range(self.model.height):
    #         for j in range(int(2*self.model.width/3), self.model.width):
    #             reward_mat[j,i] = reward/not_dead
    #             stress_mat[j,i] = stress/not_dead
                
    #     return reward_mat, stress_mat

    
    # Function appears to be deprecated. No need for initial cells parameter in model.
    # def get_fitness(self,Cell):
        
    #     fitness_score=0
    #     for i in range(self.model.height):
    #          for j in range(int(self.model.width)):
    #              if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
    #                  cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
    #                  if cell.state_tissue == cell.goal:
    #                      fitness_score+=1
                         
    #     fitness_score = fitness_score/(self.model.height*self.model.width)*100
    #                  #else:
    #                      #self.fitness-=1
                    
    #     remaining_cells = self.get_breed_count(Cell)
    #     if remaining_cells == 0:
    #         remaining_cells=1
    #     stress = self.get_cellular_stress(Cell)
    #     geometrical_frustration = self.general_geometrical_frustration(Cell)/(self.model.height*self.model.width)*100

    #     fitness = 3*fitness_score  - 10*(self.model.initial_cells -  remaining_cells)/(self.model.height*self.model.width)*100 #- 2*stress/(remaining_cells)# - 10*((np.abs(33-perc_blue) + np.abs(33-perc_white) + np.abs(33-perc_red)))

    #     return fitness
    
    def generate_cells(self, Cell, start):
        # Use start mat to generate starting cells
        for i in range(self.model.height):
            for j in range(self.model.width):
                state_cell = start[i][j]
                state_tissue = start[i][j]
                molecules =  [random.random()] #np.array([3.,3.]) #
                if state_cell == 0:
                    continue
                if state_cell== 1:
                    molecules[0] =  11
                elif state_cell== 2:
                    molecules[0] = 3
                elif state_cell== 3:
                    molecules[0] = 7
                cell = Cell(self.model.net, self.model.depth, self.model.next_id(), (j, i), self.model,  True, 
                            energy = self.model.energy, energyt1 =  self.model.energy, cell_gain_from_good_state = 0,  
                            molecules = molecules, goal = self.model.goal[i][j], GJ_opening_molecs=0, 
                            GJ_opening_stress=0, stress = 0, stresst1=0, decision_state0=0, decision_state1=0,decision_state2=0, 
                            state=state_cell, statet1=state_cell, state_tissue = state_tissue)
                self.model.grid.place_agent(cell, (j, i))
                self.model.schedule.add(cell)
                     
                     
    
                     
                     
                     
