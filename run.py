from multicellularity.model_for_analysis import Multicellularity_model
from random_faces import RandomFaces
from analysis import sim

import os
import time
import numpy as np
import MultiNEAT as NEAT
from MultiNEAT import  EvaluateGenomeList_Parallel
import datetime
import shutil
import multiprocessing
import textwrap
import pickle


def save_exp_files(parent_dir, exp_file_path):
    # Saving files and folders
    calling_file = os.path.basename(exp_file_path)
    shutil.copyfile(f"{exp_file_path}",  f"{parent_dir}/{calling_file}")
    shutil.copyfile("analysis.py",  f"{parent_dir}/analysis.py")
    shutil.copyfile("run_analysis.py",  f"{parent_dir}/run_analysis.py")
    shutil.copyfile("random_faces.py",  f"{parent_dir}/random_faces.py")
    shutil.copyfile("run.py",  f"{parent_dir}/run.py")
    shutil.copyfile("experiment.py",  f"{parent_dir}/experiment.py")
    shutil.copytree('./multicellularity', f"{parent_dir}/multicellularity")
    # shutil.copytree('./Experiments', f"{parent_dir}/Experiments")



def save_winning_results(best_genome, best_fitness, best_ID, elapsed_time, exp, parent_dir):
    # Saving score
    with open("score.txt","w+") as score_file:
        score_file.write("\nBest ever fitness: %f, genome ID: %d" % (best_fitness, best_ID))
        score_file.write("\nTrial elapsed time: %.3f sec" % (elapsed_time))
    os.replace("score.txt", f"{parent_dir}/score.txt")

    # Visualize best network's Genome
    winner_net_CPPN = NEAT.NeuralNetwork()
    best_genome.BuildPhenotype(winner_net_CPPN)
    # Visualize best network's Phenotype
    winner_net = NEAT.NeuralNetwork()
    best_genome.BuildESHyperNEATPhenotype(winner_net, exp.substrate, exp.params)
    winner_net.Save(f"{parent_dir}/winner_net.txt")

    # Pickle the experiment (not needed for sim)
    exp_file = os.path.join(".", "exp.pickle")
    with open(exp_file, 'wb') as f:
        pickle.dump(exp, f, pickle.HIGHEST_PROTOCOL)
    os.replace(exp_file, f"{parent_dir}/exp.pickle")

    return winner_net



def run_trial(exp, pop, cpu_number):
    best_genomes = []
    best_fitness = -20000
    best_ID = -1

    for generation in range(exp.nb_gens):
        gen_time = time.time()

        # Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)
        fitnesses = EvaluateGenomeList_Parallel(genome_list, eval_individual, display=False, cores=cpu_number)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]


        gen_best_genome = pop.Species[0].GetLeader()   
        gen_best_fitness = gen_best_genome.GetFitness()
        gen_best_ID = gen_best_genome.GetID()

        if (gen_best_fitness > best_fitness):
            best_genomes.append(gen_best_genome)
            best_fitness = gen_best_fitness
            best_ID = gen_best_ID

        # Advance to the next generation
        pop.Epoch()

        # Print generation's statistics    
        gen_elapsed_time = time.time() - gen_time
        print_generation_stats(gen_elapsed_time, gen_best_fitness, gen_best_ID, best_fitness, best_ID, generation)    

        solution_found = (gen_best_fitness >= exp.max_fitness) 
        if solution_found:
            break

    return best_genomes, best_fitness, best_ID



def print_generation_stats(gen_elapsed_time, gen_best_fitness, gen_best_ID, best_fitness, best_ID, generation):
    output = f"""
    *****************************************************
    GEN: {generation}

        Generation best fitness:    {round(gen_best_fitness,1)},    ID: {gen_best_ID}
        Generation elapsed time:    {round(gen_elapsed_time,1)}
        Trial best fitness so far:  {round(best_fitness,1)},    ID: {best_ID}
    *****************************************************
    """
    print(textwrap.dedent(output))



def print_trial_stats(elapsed_time, file, best_fitness, best_ID):
    # Making it easy to copy and paste to view file
    file_name = "_".join([file,str(int(best_fitness))]).replace(":", "\:")
    output = f"""
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    EVOLUTION DONE

        Trial best fitness:     {round(best_fitness,1)},    ID: {best_ID}
        Trial elapsed time:     {round(elapsed_time,1)}
        File:                   {file_name}
        Run it:                 python ../sfc_results/{file_name}/run_analysis.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    print(textwrap.dedent(output))



def eval_individual(genome):
    global g_exp
    
    net = NEAT.NeuralNetwork()
    genome.BuildESHyperNEATPhenotype(net, g_exp.substrate, g_exp.params)
    net.Flush()

    # random_start == true means that each generation should be trained on a different random face, to get a robust NN.
    if g_exp.random_start==True:
        g_exp.start = RandomFaces().get_random_face()[::-1]

    runs = 3
    fit = 0
    for i in range(runs):
        model = Multicellularity_model(
            net = net, 
            exp = g_exp,
        )
        model.verbose = False
        trial = model.run_model(fitness_evaluation=True)
        fit += trial
    fit /= runs

    return fit



def test_eval_individual(net, exp):
    
    fit = 0
    for i in range(3):
        model = Multicellularity_model(net = net, exp = exp)
        model.verbose = False
        run = model.run_model(fitness_evaluation=True)
        fit += run
        print(f"Run {i+1}: {round(run,1)}")
    fit/=3
    print(f"Expected fitness (run.py): {fit}")
    return fit



g_exp=None # global experiment

def run_experiment(exp, exp_file_path):
    """
        exp: the experiment parameters
    """
    # Updating global experiment
    cpu_number = multiprocessing.cpu_count()
    print(f"CPU cores: {cpu_number}")


    global g_exp
    g_exp = exp

    # Result file
    myDatetime = datetime.datetime.now()
    myString = myDatetime.strftime('%Y/%b/%d/%H:%M:%S')
    file = myString.replace(' ','_')
    parent_dir = f"../sfc_results/{file}"
    os.makedirs(f"{parent_dir}", exist_ok=True)
    save_exp_files(parent_dir, exp_file_path)
    

    # Save random seed and exp.params
    seed = int(time.time()) #1660341957#
    np.save(f"{parent_dir}/seed", seed)

    genome = NEAT.Genome(
        0,
        exp.substrate.GetMinCPPNInputs(),
        2,  # hidden units
        exp.substrate.GetMinCPPNOutputs(),
        False,
        NEAT.ActivationFunction.TANH,
        NEAT.ActivationFunction.SIGNED_GAUSS,
        1,  # hidden layers seed
        exp.params, 
        1   # one hidden layer
    )  
    
    pop = NEAT.Population(genome, exp.params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    # Run trial for up to N generations.
    start_time = time.time()
    best_genomes, best_fitness, best_ID = run_trial(exp, pop, cpu_number)

    ## Print trial statistics
    elapsed_time = time.time() - start_time
    print_trial_stats(elapsed_time, file, best_fitness, best_ID)

    # Match expectations?
    best_genome = best_genomes[-1]

    # Saving winning results
    winner_net = save_winning_results(best_genome, best_fitness, best_ID, elapsed_time, exp, parent_dir)

    # Testing the winner to see if authentic
    net = NEAT.NeuralNetwork()
    net.Load(f"{parent_dir}/winner_net.txt")
    test_eval_individual(net, exp)

    # Adding fitness to file name
    formatted_fit = str(int(best_fitness))
    os.rename(f"{parent_dir}", f"{parent_dir}_{formatted_fit}" ) # used to be "Results/" for both

    # Simulate the results if option turned on
    if exp.simulate == True:
        sim(exp, winner_net)
        
    return best_fitness, elapsed_time, f"{file}_{formatted_fit}"


