import MultiNEAT as NEAT
import pickle
import sys


class experiment:
    # MESA params
    nb_gens=2
    depth=4
    max_fitness= 97
    energy = 70             # default was 70
    step_count = 100
    fitness_function = ""   # see model class for the definitions of the variable
    nb_gap_junctions = 1
    nb_stress_GJ = 1
    nb_output_molecules = 1 
    nb_output_stress = 1
    nb_output_anxio = 1
    apoptosis_on = 1        # Set to 0 if off
    cell_division_on = 1    # Set to 0 if off

    # Interface
    interface =  False

    # Parameters for es-hyperneat
    params = NEAT.Parameters()
    params.PopulationSize = 350
    params.DynamicCompatibility = True
    params.CompatTreshold = 3.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 10
    params.OldAgeTreshold = 35
    params.StagnationDelta = 5
    params.MinSpecies = 5
    params.MaxSpecies = 15
    params.RouletteWheelSelection = False
    params.MutateRemLinkProb = 0.02
    params.RecurrentProb = 0.2
    params.OverallMutationRate = 0.15
    params.MutateAddLinkProb = 0.03
    params.MutateAddNeuronProb = 0.03
    params.MutateWeightsProb = 0.90
    params.MaxWeight = 8.0
    params.MinWeight = -8.0
    params.WeightMutationMaxPower = 0.2
    params.WeightReplacementMaxPower = 1.0
    params.MutateActivationAProb = 0.0
    params.ActivationAMutationMaxPower = 0.5
    params.MinActivationA = 0.05
    params.MaxActivationA = 6.9 
    params.MinNeuronBias = -params.MaxWeight
    params.MaxNeuronBias = params.MaxWeight
    params.MutateNeuronActivationTypeProb = 0.3
    params.ActivationFunction_SignedGauss_Prob = 1.0
    params.ActivationFunction_SignedStep_Prob = 1.0
    params.ActivationFunction_Linear_Prob = 1.0
    params.ActivationFunction_SignedSine_Prob = 1.0
    params.ActivationFunction_SignedSigmoid_Prob = 1.0
    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    params.ActivationFunction_TanhCubic_Prob = 0.0
    params.ActivationFunction_UnsignedStep_Prob = 0.0
    params.ActivationFunction_UnsignedGauss_Prob = 0.0
    params.ActivationFunction_Abs_Prob = 0.0
    params.ActivationFunction_UnsignedSine_Prob = 0.0
    params.AllowLoops = True
    params.AllowClones = True
    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0
    params.DivisionThreshold = 0.03 
    params.VarianceThreshold = 0.03
    params.BandThreshold = 0.3
    # depth of the quadtree
    params.InitialDepth = 3
    params.MaxDepth = 3
    # corresponds to the number of hidden layers = iterationlevel+1
    params.IterationLevel = depth -1
    params.Leo = False
    params.GeometrySeed = True
    params.LeoSeed = True
    params.LeoThreshold = 0.2
    params.CPPN_Bias = -1.0
    params.Qtree_X = 0.0
    params.Qtree_Y = 0.0
    params.Width = 1.
    params.Height = 1.
    params.Elitism = 0.1
    rng = NEAT.RNG()
    rng.TimeSeed()

    nb_inputs = 10 # nb molecules + energy + stress + state + bias + state_neigbours
    # The "1" represents the one GJ_opening_molecs variable that will exist. As of now, there won't be one of those variables for each molecule.
    nb_outputs =  nb_output_molecules + 1 + nb_output_stress + 1 + nb_output_anxio + apoptosis_on + cell_division_on
    # output_tags = ["m0_to_send", "GJ_opening_molecs", "stress_to_send", "stress_GJ_opening", "anxio_to_send", "apoptosis", "cell_division"]

    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        # self.model_type = model_type
        self.height=len(goal)
        self.width=len(goal)
        self.initial_cells=self.height*self.width

        #  Substrate for MultiNEAT
        self.input_coordinates = [(-1. +(2.*i/(self.nb_inputs - 1)), -1.) for i in range(0, self.nb_inputs)]
        self.output_coordinates = [(-1. +(2.*i/(self.nb_outputs - 1)),1.) for i in range(0, self.nb_outputs)]
        self.substrate = NEAT.Substrate(self.input_coordinates, [],  self.output_coordinates)
        self.substrate.m_allow_input_hidden_links = True
        self.substrate.m_allow_hidden_output_links = True
        self.substrate.m_allow_hidden_hidden_links = False
        self.substrate.m_allow_looped_hidden_links = True
        self.substrate.m_allow_input_output_links = False
        self.substrate.m_allow_output_hidden_links = True
        self.substrate.m_allow_output_output_links = False
        self.substrate.m_allow_looped_output_links = False
        self.substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
        self.substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID
        self.substrate.m_with_distance = True
        self.substrate.m_max_weight_and_bias = 8.0
        try:
            x = pickle.dumps(self.substrate)
        except:
            print('You have mistyped a substrate member name upon setup. Please fix it.')
            sys.exit(1)