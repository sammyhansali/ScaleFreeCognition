import MultiNEAT as NEAT
import pickle
import sys


class experiment:
    # MESA params
    nb_gens=2
    depth = 4
    max_fitness= 98
    energy = 50            # default was 70
    step_count = 100
    # fitness = 0           # see model class for the definitions of the variable
    nb_gap_junctions = 1    # one unique GJ_opening for both molecs and stress
    nb_stress_GJ = 1
    nb_output_stress = 1
    nb_output_anxio = 1
    apoptosis_on = 1        # Set to 0 if off
    cell_division_on = 1    # Set to 0 if off
    history_length = 5
    ANN_inputs = None
    ANN_outputs = None
    e_penalty = None
    random_start = False
    preset = None
    multiple = 5

    # Interface
    interface =  False

    # Parameters for es-hyperneat
    params = NEAT.Parameters()
    params.PopulationSize = 70
    # params.PopulationSize = 10
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

    nb_ANN_inputs = 0 
    nb_ANN_outputs = 0

    bioelectric_stimulus =  [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def __init__(   self, 
                    start, 
                    goal, 
                    ANN_inputs,
                    nb_ANN_inputs,
                    ANN_outputs,
                    history_length,
                    nb_gens,
                    e_penalty,
                    nb_output_molecules,
                ):
        self.ANN_inputs = ANN_inputs
        self.ANN_outputs = ANN_outputs
        # self.nb_ANN_inputs = len(self.ANN_inputs)*history_length
        self.nb_ANN_inputs = nb_ANN_inputs
        self.nb_ANN_outputs = len(self.ANN_outputs)
        self.history_length = history_length
        self.nb_gens = nb_gens
        self.e_penalty = e_penalty

        self.start = start
        self.goal = goal
        # self.model_type = model_type
        self.height=len(goal)
        self.width=len(goal)
        # self.initial_cells=self.height*self.width
        self.nb_output_molecules = nb_output_molecules

        #  Substrate for MultiNEAT
        self.input_coordinates = [(-1. +(2.*i/(self.nb_ANN_inputs - 1)), -1.) for i in range(0, self.nb_ANN_inputs)]
        self.output_coordinates = [(-1. +(2.*i/(self.nb_ANN_outputs - 1)),1.) for i in range(0, self.nb_ANN_outputs)]
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