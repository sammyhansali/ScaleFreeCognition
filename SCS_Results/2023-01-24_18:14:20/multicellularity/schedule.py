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
        nb_eye_goal = 6
        nb_eye_matched = 0

        nb_nose_mouth_goal = 7
        nb_nose_mouth_matched = 0

        nb_skin_goal = 68
        nb_skin_matched = 0

        for i in range(self.model.height):
            for j in range(int(self.model.width)):

                if len(self.model.grid.get_cell_list_contents([(j,i)]))>0:
                    cell = self.model.grid.get_cell_list_contents([(j,i)])[0]
                    if cell.cell_type[0] == cell.goal_cell_type:
                        if cell_type[0] == '3':
                            nb_eye_matched+=1
                        elif cell_type[0] == '2':
                            nb_nose_mouth_matched+=1
                        elif cell_type[0] == '1':
                            nb_skin_matched+=1
        
        # Ideally you make more rewards with lower value (reward 10 for 1 eye, 10 for 2 eyes, etc.)
        if nb_eye_goal==nb_eye_matched:
            fitness_score+=20
        elif nb_nose_mouth_goal==nb_nose_mouth_matched:
            fitness_score+=20
        elif nb_skin_goal==nb_skin_matched:
            fitness_score+=10
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


    def generate_cells(self, Cell):
        # Use start mat to generate starting cells
        for y in range(self.model.height):
            for x in range(self.model.width):
                
                cell_type = self.model.start[y][x]
                potential = self.model.bioelectric_stimulus[y][x] # Cell's potential won't change

                # Setting up communication network
                molecules = {}
                GJ_molecules = {}
                for n in range(self.model.nb_output_molecules):
                    molecules[n] = [0]*self.model.history_length
                    amount = 0
                    if cell_type == 1:
                        amount = 0
                    elif cell_type == 2:
                        amount = 5
                    elif cell_type == 3:
                        amount = 10
                    # elif cell_type == 4:
                    #     amount = 15
                    molecules[n][0] = amount     # Arbitrary amount of signaling molecules to begin with
                    molecules[n][1] = amount     # Arbitrary amount of signaling molecules to begin with
                    GJ_molecules[n] = 0

                cell = Cell(    net = self.model.net, 
                                depth = self.model.depth, 
                                unique_id = self.model.next_id(), 
                                pos = (x, y), 
                                model = self.model,  
                                moore = True, 
                                goal_cell_type = self.model.goal[y][x], 
                                bioelectric_stimulus = None, 
                                GJ_opening_ions=0, 
                                # GJ_opening_molecs=0, 
                                GJ_opening_stress=0, 
                                GJ_molecules = GJ_molecules,
                                # Historical data
                                energy = [0]*self.model.history_length,
                                stress = [0]*self.model.history_length, 
                                # global_fitness = [0]*self.model.history_length,
                                direction = [0]*self.model.history_length,
                                cell_type = [0]*self.model.history_length,
                                potential = [0]*self.model.history_length,
                                molecules = molecules, 
                            )
                cell.energy[0] = self.model.energy
                cell.direction[0] = random.choice(range(1,9))
                cell.cell_type[0] = cell_type
                cell.potential[0] = potential
                # cell.potential[0] = resting_pot
                
                self.model.grid.place_agent(cell, (x, y))
                self.model.schedule.add(cell)
    
                     
                
                     
