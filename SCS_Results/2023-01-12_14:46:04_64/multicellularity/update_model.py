import random

class UpdateModel:
    # Default, run the simulation as normal.
    def preset1(self, model):
        return

    # Turn electric face over after time step 100.
    # def preset_bioelectric(self, model):
    #     if model.schedule.time == 100: 

    #         bioelectric_stimulus =      [[ -80, -80, -80, -80, -80, -80, -80, -80, -80],
    #                                     [ -80, -80, -80, -80, -80, -10, -10, -80, -80],
    #                                     [ -80, -50, -80, -80, -80, -80, -10, -80, -80],
    #                                     [ -80, -50, -80, -80, -50, -80, -80, -80, -80],
    #                                     [ -80, -50, -80, -80, -80, -80, -80, -80, -80],
    #                                     [ -80, -50, -80, -80, -50, -80, -80, -80, -80],
    #                                     [ -80, -50, -80, -80, -80, -80, -10, -80, -80],
    #                                     [ -80, -50, -80, -80, -80, -10, -10, -80, -80],
    #                                     [ -80, -80, -80, -80, -80, -80, -80, -80, -80]]
    #         model.bioelectric_stimulus = bioelectric_stimulus[::-1]

    # # Make neutral after x amount of steps. Do the cells remember intended pattern?
    # def preset_bioelectric_transient(self, model):
    #     if model.schedule.time == 25: 

    #         bioelectric_stimulus =      [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    #         model.bioelectric_stimulus = bioelectric_stimulus[::-1]

    # # Set molecule x equal to zero after time 100.
    # def preset_remove_molecule(self, model):
    #     if model.schedule.time == 10: 
            
    #         molecule_to_remove = random.choice(range(model.nb_output_molecules))
    #         model.schedule.remove_molecule(molecule_to_remove)
    #         model.schedule.reset_energy(model.energy)

    def preset_bioelectric(self, model, time):
        if model.schedule.time == time: 

            bioelectric_stimulus =      [[ -80, -80, -80, -80, -80, -80, -80, -80, -80],
                                        [ -80, -80, -80, -80, -80, -10, -10, -80, -80],
                                        [ -80, -50, -80, -80, -80, -80, -10, -80, -80],
                                        [ -80, -50, -80, -80, -50, -80, -80, -80, -80],
                                        [ -80, -50, -80, -80, -80, -80, -80, -80, -80],
                                        [ -80, -50, -80, -80, -50, -80, -80, -80, -80],
                                        [ -80, -50, -80, -80, -80, -80, -10, -80, -80],
                                        [ -80, -50, -80, -80, -80, -10, -10, -80, -80],
                                        [ -80, -80, -80, -80, -80, -80, -80, -80, -80]]
            model.bioelectric_stimulus = bioelectric_stimulus[::-1]

    def preset_remove_molecule(self, model, arr):
        time = arr[0]
        molecule_to_remove = arr[1]

        if model.schedule.time == time: 
            model.schedule.remove_molecule(molecule_to_remove)

    def preset_change_gap_junctions(self, model):
        return