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
        # self.molecules_exchanged = [0]

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

        # Update energy of each cell
        agent_keys = list(self._agents.keys())
        for key in agent_keys:
            cell = self._agents[key]
            cell.energy = cell.update_history(cell.energy, cell.energy_temp + cell.model.global_fitness[0] - cell.model.e_penalty)



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
            if self.agents_by_breed[Cell][agent_key].cell_type[0] != self.agents_by_breed[Cell][agent_key].goal_cell_type:
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

    # def get_nb_molecules_exchanged(self, Cell, i):
    #     # subtract molecules[i][j] and molecules[i][j-1] for each agent
    #     n = self.model.nb_output_molecules
    #     # Assume n > 0
    #     # if n = 1, return exchanged for molec_0 only
    #     # if n = 2, return exchanged for molec_0, molec_1
    #     # Etc
    #     # return [n,n]
    #     # exchanged = [0]*n
    #     # agent_keys = list(self.agents_by_breed[Cell].keys())
    #     # # print(agent_keys)
    #     # for agent_key in agent_keys:
    #     #     for i in range(n):
    #     #         curr= self.agents_by_breed[Cell][agent_key].molecules[i][0]
    #     #         prev= self.agents_by_breed[Cell][agent_key].molecules[i][1]
    #     #         # return [curr, prev]
    #     #         exchanged[i] += abs(curr - prev)

    #     # return [x/2 for x in exchanged]
    #     # self.molecules_exchanged = [x/2 for x in exchanged]

    #     exchanged = 0
    #     agent_keys = list(self.agents_by_breed[Cell].keys())
    #     if i == "total":
    #         for agent_key in agent_keys:
    #             curr=0
    #             prev=0
    #             for x in range(n):
    #                 curr += self.agents_by_breed[Cell][agent_key].molecules[x][0]
    #                 prev += self.agents_by_breed[Cell][agent_key].molecules[x][1]
    #                 # return [curr, prev]
    #             exchanged += abs(curr - prev)
    #     else:
    #         for agent_key in agent_keys:
    #             curr = self.agents_by_breed[Cell][agent_key].molecules[i][0]
    #             prev = self.agents_by_breed[Cell][agent_key].molecules[i][1]
    #             # return [curr, prev]
    #             exchanged += abs(curr - prev)

    #     return exchanged/2

    # def get_nb_molecules_exchanged(self, Cell, i):
    #     if i == "total":
    #         return sum(self.molecules_exchanged)
    #     # Otherwise, i is an integer that picks which molecule you want
    #     return self.molecules_exchanged[i]
             
    def get_nb_molecules_exchanged_0(self, Cell):
        # subtract molecules[i][j] and molecules[i][j-1] for each agent
        n = self.model.nb_output_molecules
        exchanged = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())

        for agent_key in agent_keys:
            curr = self.agents_by_breed[Cell][agent_key].molecules[0][0]
            prev = self.agents_by_breed[Cell][agent_key].molecules[0][1]
            # return [curr, prev]
            exchanged += abs(curr - prev)
        
        return exchanged/2

    def get_nb_molecules_exchanged_1(self, Cell):
        # subtract molecules[i][j] and molecules[i][j-1] for each agent
        n = self.model.nb_output_molecules
        exchanged = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())

        for agent_key in agent_keys:
            curr = self.agents_by_breed[Cell][agent_key].molecules[1][0]
            prev = self.agents_by_breed[Cell][agent_key].molecules[1][1]
            # return [curr, prev]
            exchanged += abs(curr - prev)
        
        return exchanged/2

    def get_nb_molecules_exchanged_2(self, Cell):
        # subtract molecules[i][j] and molecules[i][j-1] for each agent
        n = self.model.nb_output_molecules
        exchanged = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())

        for agent_key in agent_keys:
            curr = self.agents_by_breed[Cell][agent_key].molecules[2][0]
            prev = self.agents_by_breed[Cell][agent_key].molecules[2][1]
            # return [curr, prev]
            exchanged += abs(curr - prev)
        
        return exchanged/2

    def get_nb_molecules_exchanged_3(self, Cell):
        # subtract molecules[i][j] and molecules[i][j-1] for each agent
        n = self.model.nb_output_molecules
        exchanged = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())

        for agent_key in agent_keys:
            curr = self.agents_by_breed[Cell][agent_key].molecules[3][0]
            prev = self.agents_by_breed[Cell][agent_key].molecules[3][1]
            # return [curr, prev]
            exchanged += abs(curr - prev)
        
        return exchanged/2

    def get_nb_molecules_exchanged_4(self, Cell):
        # subtract molecules[i][j] and molecules[i][j-1] for each agent
        n = self.model.nb_output_molecules
        exchanged = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())

        for agent_key in agent_keys:
            curr = self.agents_by_breed[Cell][agent_key].molecules[4][0]
            prev = self.agents_by_breed[Cell][agent_key].molecules[4][1]
            # return [curr, prev]
            exchanged += abs(curr - prev)
        
        return exchanged/2

    def get_nb_molecules_exchanged_5(self, Cell):
        # subtract molecules[i][j] and molecules[i][j-1] for each agent
        n = self.model.nb_output_molecules
        exchanged = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())

        for agent_key in agent_keys:
            curr = self.agents_by_breed[Cell][agent_key].molecules[5][0]
            prev = self.agents_by_breed[Cell][agent_key].molecules[5][1]
            # return [curr, prev]
            exchanged += abs(curr - prev)
        
        return exchanged/2

    def get_nb_molecules_exchanged_tot(self, Cell):
        # subtract molecules[i][j] and molecules[i][j-1] for each agent
        n = self.model.nb_output_molecules
        exchanged = 0
        agent_keys = list(self.agents_by_breed[Cell].keys())

        for agent_key in agent_keys:
            curr=0
            prev=0
            for x in range(n):
                curr += self.agents_by_breed[Cell][agent_key].molecules[x][0]
                prev += self.agents_by_breed[Cell][agent_key].molecules[x][1]
            exchanged += abs(curr - prev)
        
        return exchanged/2
        
    def global_fitness(self):
        curr_dissimilarity = self.dissimilarity()
        
        fitness_score = 100*(1-curr_dissimilarity/self.model.start_dissimilarity) 
        # fitness_score/(self.model.height*self.model.width)*100
        return fitness_score
    
    def dissimilarity(self):
        curr_dissimilarity=0
        for i in range(self.model.height):
            for j in range(int(self.model.width)):
                # alive cell on spot
                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                    if cell.cell_type[0] != cell.goal_cell_type:
                        curr_dissimilarity += 1
                # nothing on spot
                else:
                    if self.model.goal[i][j] != 0:
                        curr_dissimilarity += 1
        return curr_dissimilarity

    def organ_focused_fitness(self):
        
        nb_skin_goal = 68
        nb_skin_matched = 0
        
        nb_eye_goal = 6
        nb_eye_matched = 0

        # nb_nose_mouth_goal = 7
        # nb_nose_mouth_matched = 0
        nb_mouth_goal = 5
        nb_mouth_matched = 0

        nb_nose_goal = 2
        nb_nose_matched = 0


        for i in range(self.model.height):
            for j in range(int(self.model.width)):

                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                    if cell.cell_type[0] == cell.goal_cell_type:
                        if cell.cell_type[0] == 3:
                            nb_eye_matched+=1
                        elif cell.cell_type[0] == 2:
                            # nb_nose_mouth_matched+=1
                            nb_mouth_matched+=1
                        elif cell.cell_type[0] == 1:
                            nb_skin_matched+=1
                        # New-for 4 cell types
                        elif cell.cell_type[0] == 4:
                            nb_nose_matched+=1

        ### Base scores
        eye_base_score = 20*(nb_eye_matched/nb_eye_goal) # Max score of 20
        # nose_mouth_base_score = 20*(nb_nose_mouth_matched/nb_nose_mouth_goal) # Max score of 20
        mouth_base_score = 18*(nb_mouth_matched/nb_mouth_goal) # Max score of 20
        nose_base_score = 4*(nb_nose_matched/nb_nose_goal) # Max score of 20
        skin_base_score = 36*(nb_skin_matched/nb_skin_goal) # Max score of 20
        # fitness_score = eye_base_score + nose_mouth_base_score + skin_base_score
        fitness_score = eye_base_score + mouth_base_score + nose_base_score + skin_base_score

        ### Bonuses for completing organs
        # Eyes
        if nb_eye_matched==nb_eye_goal:
            fitness_score+=10
        elif nb_eye_matched >= nb_eye_goal/2:       # Microreward
            fitness_score+=5
        
        # Mouth
        if nb_mouth_matched==nb_mouth_goal:
            fitness_score+=10
        elif nb_mouth_matched >= nb_mouth_goal/2:   # Microreward
            fitness_score+=5

        # Nose
        if nb_nose_matched==nb_nose_goal:
            fitness_score+=2

        return fitness_score

    def french_flag_fitness(self):

        nb_left_goal = 27
        nb_left_matched = 0
        
        nb_middle_goal = 27
        nb_middle_matched = 0
        
        nb_right_goal = 27
        nb_right_matched = 0

        for i in range(self.model.height):
            for j in range(int(self.model.width)):

                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                    if cell.cell_type[0] == cell.goal_cell_type:
                        if cell.cell_type[0] == 3:
                            nb_left_matched+=1
                        elif cell.cell_type[0] == 1:
                            nb_middle_matched+=1
                        elif cell.cell_type[0] == 2:
                            nb_right_matched+=1

        ### Base scores
        left_base_score = 25*(nb_left_matched/nb_left_goal) # Max score of 20
        middle_base_score = 25*(nb_middle_matched/nb_middle_goal) # Max score of 20
        right_base_score = 25*(nb_right_matched/nb_right_goal) # Max score of 20

        fitness_score = left_base_score + middle_base_score + right_base_score

        ### Bonuses for completing organs
        # Left
        if nb_left_matched==nb_left_goal:
            fitness_score+=8
        elif nb_left_matched >= nb_left_goal/2:       # Microreward
            fitness_score+=5
        
        # Middle
        if nb_middle_matched==nb_middle_goal:
            fitness_score+=8
        elif nb_middle_matched >= nb_middle_goal/2:       # Microreward
            fitness_score+=5

        # Right
        if nb_right_matched==nb_right_goal:
            fitness_score+=8
        elif nb_right_matched >= nb_right_goal/2:       # Microreward
            fitness_score+=5

        return fitness_score

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

    def remove_molecule(self, molecule_to_remove):
        for y in range(self.model.height):
            for x in range(int(self.model.width)):
                # alive cell on spot
                if len(self.model.grid.get_cell_list_contents([(x,y)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(x,y)])[0]
                    cell.molecules[molecule_to_remove] = cell.update_history(cell.molecules[molecule_to_remove], 0)

    def remove_molecule_and_add(self, molecule_to_remove):
        for y in range(self.model.height):
            for x in range(int(self.model.width)):
                # alive cell on spot
                if len(self.model.grid.get_cell_list_contents([(x,y)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(x,y)])[0]
                    n=self.model.nb_output_molecules
                    add_amount = cell.molecules[molecule_to_remove][0]/(n-1)
                    for i in range(n):
                        if i == molecule_to_remove:
                            cell.molecules[i] = cell.update_history(cell.molecules[i], 0)
                        else:
                            cell.molecules[i] = cell.update_history(cell.molecules[i], cell.molecules[i][0]+add_amount)

    def reset_energy(self, amount):
        for y in range(self.model.height):
            for x in range(int(self.model.width)):
                # alive cell on spot
                if len(self.model.grid.get_cell_list_contents([(x,y)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(x,y)])[0]
                    cell.energy = cell.update_history(cell.energy, amount)

    def change_gap_junctions(self, GJ_nb, new_GJ):
        for y in range(self.model.height):
            for x in range(int(self.model.width)):
                # alive cell on spot
                if len(self.model.grid.get_cell_list_contents([(x,y)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(x,y)])[0]
                    cell.GJ_molecules[GJ_nb] = new_GJ[y][x] # Add history later


    def generate_cells(self, Cell, start_molecs):
        # Use start mat to generate starting cells
        for y in range(self.model.height):
            for x in range(self.model.width):
                
                cell_type = self.model.start[y][x]
                potential = self.model.bioelectric_stimulus[y][x] # Cell's potential won't change

                molecules = {n: [0] * self.model.history_length for n in range(self.model.nb_output_molecules)}
                GJ_molecules = {n: 0 for n in range(self.model.nb_output_molecules)}

                # # Totaling OR agent decides when to diff
                # if self.model.ef_mode == 1:
                #     for n in range(self.model.nb_output_molecules):
                #         amount = start_molecs[cell_type]
                #         molecules[n][0] = amount     
                #         molecules[n][1] = amount     
                # # elif self.model.ef_mode == 2 or self.model.ef_mode == 3 or self.model.ef_mode == 4:
                # else:
                #     for n in range(self.model.nb_output_molecules):
                #         # Molec 0 corresps to cell type 1, molec 1 to cell type 2, etc. Rest of molec classes will have 2.5 each
                #         amount = start_molecs[cell_type]
                #         if n == cell_type-1:
                #             amount += 2.5
                #         molecules[n][0] = amount     
                #         molecules[n][1] = amount     

                ## Adding correct initial molecule amounts
                for n in range(self.model.nb_output_molecules):
                    # Molec 0 corresps to cell type 1, molec 1 to cell type 2, etc
                    amount = start_molecs[cell_type]
                    if (not self.model.ef_mode == 1) and (n == cell_type-1):
                        amount += 2.5
                    if (self.model.ef_mode==4 or self.model.ef_mode==6) and (n == cell_type-1): 
                        # Conditional exists JUST so my sbatch expers from 3/10 dont interfere with sbatch exper (mode 5s) from 3/9 that are still running
                        amount = 5
                    molecules[n][0] = amount     
                    molecules[n][1] = amount  

                ## Initializing the cell
                cell = Cell(    
                    net = self.model.net, 
                    depth = self.model.depth, 
                    unique_id = self.model.next_id(), 
                    pos = (x, y), 
                    model = self.model,  
                    moore = True, 
                    goal_cell_type = self.model.goal[y][x], 
                    bioelectric_stimulus = None, 
                    GJ_opening_ions=0, 
                    GJ_opening_stress=0, 
                    GJ_molecules = GJ_molecules,
                    # Historical data
                    energy = [0]*self.model.history_length,
                    stress = [0]*self.model.history_length, 
                    global_fitness = self.model.global_fitness,
                    direction = [0]*self.model.history_length,
                    cell_type = [0]*self.model.history_length,
                    potential = [0]*self.model.history_length,
                    molecules = molecules, 
                )
                cell.energy[0] = self.model.energy
                cell.direction[0] = random.choice(range(1,9))
                cell.cell_type[0] = cell_type
                cell.potential[0] = potential
                
                ## Placing agent in grid and schedule
                self.model.grid.place_agent(cell, (x, y))
                self.model.schedule.add(cell)
    
                     
                
                     
