import random

class UpdateModel:
    # Default, run the simulation as normal.
    def preset1(self, model):
        return

    def preset_bioelectric(self, model, arr):
        time, new_stim = arr

        if model.schedule.time == time: 
            model.bioelectric_stimulus = new_stim

    def preset_goal(self, model, arr):
        time, new_goal = arr

        if model.schedule.time == time: 
            model.goal = new_goal

    def preset_remove_molecule(self, model, arr):
        time, molecule_to_remove = arr

        if model.schedule.time == time: 
            model.schedule.remove_molecule(molecule_to_remove)
    
    def preset_reset_energy(self, model, arr):
        time, reset_to = arr

        if model.schedule.time == time: 
            model.schedule.reset_energy(reset_to)

    def preset_change_gap_junctions(self, model, arr):
        time, GJ_nb, new_GJ = arr

        if model.schedule.time == time:
            model.schedule.change_gap_junctions(GJ_nb, new_GJ)