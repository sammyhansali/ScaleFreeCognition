
class UpdateModel:
    def preset1(self, model):
        if model.schedule.time == 100: 

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

    # def preset2(self, model):
    #     model.variable1 = new_value3
    #     model.variable2 = new_value4