import pickle
import MultiNEAT as NEAT
from analysis import sim
from multicellularity.model_for_analysis import Multicellularity_model
from experiment import experiment

def eval_individual(exp, genome):
    
    """
    Evaluate fitness of the individual CPPN genome by creating
    the substrate with topology based on the CPPN output.
    Arguments:
        genome:         The CPPN genome
        substrate:      The substrate to build control ANN
        params:         The ES-HyperNEAT hyper-parameters
    Returns:
        fitness_individual The fitness of the individual
    """
    net = NEAT.NeuralNetwork()
    genome.BuildESHyperNEATPhenotype(net, exp.substrate, exp.params)
    net.Flush()

    # random_start == true means that each generation should be trained on a different random face, to get a robust NN.
    if exp.random_start==True:
        exp.start = RandomFaces().get_random_face()

    fit = 0
    for i in range(5):
        # model = Multicellularity_model(net = net, exp = exp)
        model = Multicellularity_model(
            net = net, 
            exp = exp,
        )
        model.verbose = False
        trial = model.run_model(fitness_evaluation=True)
        fit += trial
    fit /= 5

    return fit

# with open("/cluster/tufts/levinlab/shansa01/ScaleFreeCognition/SCS_Results/2023-02-19_08:37:56_99/exp.pickle", "rb") as fp:
#     exp = pickle.load(fp)

### Instead of importing exp, lets make own
nb_gens = 250              # 250
history_length = 2       # 2
nb_output_molecules = 4  # 3
e_penalty = 0.95        # 0.95, or 1.25 for min fit of 0 scheme
multiple = 5                    #5 for now, since reverting back to 3 cell types

## Deranged Face
start =    [[ 3, 3, 1, 1, 1, 1, 1, 1, 1],
            [ 3, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 2, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 2, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 2, 2, 2, 2, 2, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 3, 3],
            [ 1, 1, 1, 1, 1, 1, 1, 3, 1]]
start = start[::-1]     # Needed for visualization since grid placement at 0,0 is bottom left instead of top left
## Target Face
goal =     [[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 3, 3, 1, 1, 1, 3, 3, 1],
            [ 1, 3, 1, 1, 1, 1, 1, 3, 1],
            [ 1, 1, 1, 2, 1, 2, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [ 1, 1, 2, 2, 2, 2, 2, 1, 1],
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1]]
goal = goal[::-1]     # Needed for visualization since grid placement at 0,0 is bottom left instead of top left

electric = [[ -90, -90, -90, -90, -90, -90, -90, -90, -90],
            [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
            [ -90, -15, -15, -90, -90, -90, -15, -15, -90],
            [ -90, -15, -90, -90, -90, -90, -90, -15, -90],
            [ -90, -90, -90, -65, -90, -65, -90, -90, -90],
            [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
            [ -90, -90, -90, -90, -90, -90, -90, -90, -90],
            [ -90, -90, -65, -65, -65, -65, -65, -90, -90],
            [ -90, -90, -90, -90, -90, -90, -90, -90, -90]]
            # Once go back to 4 cell types, make nose -40
electric = electric[::-1]
 
ANN_inputs=     {       
                        "potential": history_length,
                        "cell_type": history_length,
                        "energy": history_length,
                        "molecules": nb_output_molecules*history_length,
                }
# nb_ANN_inputs = sum(ANN_inputs.values())

ANN_outputs=    [] 
for i in range(nb_output_molecules):
        ANN_outputs.append(f"molecule_{i}_to_send")
        ANN_outputs.append(f"molecule_{i}_GJ")
exp = experiment(
                        start, 
                        goal, 
                        ANN_inputs, 
                        # nb_ANN_inputs, 
                        ANN_outputs, 
                        history_length, 
                        nb_gens, 
                        e_penalty, 
                        nb_output_molecules
                )
if "potential" in ANN_inputs:
        exp.bioelectric_stimulus = electric

exp.multiple = multiple
# exp.random_start = True # Turned off for now
exp.start_molecs =      {
                                1:    5,
                                2:    5,
                                3:    5,
                        }

### - Even after making own, it is still random. Genome is the problem then?

# best_genome = NEAT.Genome("/cluster/tufts/levinlab/shansa01/ScaleFreeCognition/SCS_Results/2023-02-19_08:37:56_99/best_genome.txt")
with open("/cluster/tufts/levinlab/shansa01/ScaleFreeCognition/SCS_Results/2023-02-19_08:37:56_99/best_genome.pickle", "rb") as bg:
    best_genome = pickle.load(bg)

# net = NEAT.NeuralNetwork()
# net.Load("/cluster/tufts/levinlab/shansa01/ScaleFreeCognition/SCS_Results/2023-02-19_08:37:56_99/winner_net.txt")
# net.Flush()

print("eval_individ", eval_individual(exp, best_genome))
sim(exp, best_genome)
# sim(exp, net)

