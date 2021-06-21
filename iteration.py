
'''
Applies iterated learning
'''

import argparse
import os
import color_pop as cpop
import partition as part
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plot_prev_gens

# Compute convexities of previous parents chains
# and save them as .pkl


def compute_convexities(folder, round_n):
    try:
        os.mkdir(folder + "/prev_convexities")
    except:
        pass

    with open(folder + "/last_generation_" + str(round_n) + ".pkl", "rb") as f:
        parent_generation = pickle.load(f)

    agent = parent_generation.agents[0]
    generations = [*range(len(agent.prev_generations))]
    prev_convexities = []

    for color_space in agent.prev_generations:
        prev_convexities.append(color_space.degree_of_convexity())

        with open(folder + "/prev_convexities/prev_convexities_" + str(round_n) + ".pkl", "wb") as f:
            pickle.dump(prev_convexities, f)

# Generate color partition of first generation


def generate_color_partitions(n_agents=10, smooth=1, conv=1, num_labels=7, epochs=6, structure=1):
    # l is the length of the bit strings
    # returns an array with a row for each model

    # global parameters
    axis_stride = 0.05
    lab_points = part.generate_CIELab_space(axis_stride=axis_stride)
    input_size = lab_points.shape[1]
    partition_size = int(lab_points.shape[0] / num_labels)

    partitions = []

    for agent in range(n_agents):
        the_partition = part.Partition(lab_points, num_labels, np.zeros(input_size), conv,
                                       smooth)

        # Overwrite the partitions with a structured partition
        if structure:
            for label in the_partition.partition:
                for pt in range(partition_size * label, partition_size * (label + 1)):
                    the_partition.relabel_point(pt, label)

        partitions.append(the_partition)

    return partitions


def iterate(n_generations, n_agents, bottleneck, num_categories, s, c,
            save_path=False, num_trial=None, num_epochs=1, shuffle_input=False,
            output_folder=None, round_n=0):

    try:
        os.mkdir(output_folder)
    except:
        pass

    # Create first generation
    partitions = generate_color_partitions(n_agents, smooth=s, conv=c)

    parent_generation = cpop.Population(n_agents, num_categories, partitions)

    average_convexity = []
    generations = []

    with open(output_folder + '/first_gen_' + str(round_n) + '.pkl', 'wb') as f:
        pickle.dump(parent_generation, f)

    for n in range(0, n_generations):
        generations.append(n)

        # # Save number of generations that are done
        # with open(output_folder + '/generations_' + str(round_n) + '.pkl', 'wb') as f:
        #     pickle.dump(generations, f)

        # # Save the entire last generation that has been done
        with open(output_folder + '/generation_' + str(n) + "_" + str(round_n) + '.pkl', 'wb') as f:
            pickle.dump(parent_generation, f)

        # Save the final generation
        with open(output_folder + '/last_generation_' + str(round_n) + '.pkl', 'wb') as f:
            pickle.dump(parent_generation, f)

        # the new generation is created

        child_generation = cpop.Population(
            n_agents, num_categories, partitions)

        # the new generation learns from the old one
        parents = child_generation.learn_from_population(parent_generation,
                                                         bottleneck,
                                                         num_epochs,
                                                         shuffle_input)

        # the new generation becomes the old generation, ready to train the next generation
        parent_generation = child_generation
        print("Done generation {} out of {} \n\n".format(n, n_generations))

    # Save the final generation
    with open(output_folder + '/last_generation_' + str(round_n) + '.pkl', 'wb') as f:
        pickle.dump(parent_generation, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--round_n", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bottleneck", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--s", type=float, default=0.0)
    parser.add_argument("--structure", type=float, default=1.0)
    args = parser.parse_args()
    folder = "trial_" + str(args.bottleneck) + "_r" + \
        str(args.round_n) + "_epoch" + str(args.epochs) + \
        "_c" + str(args.c) + "_s" + str(args.s)
    iterate(args.iterations, 10, args.bottleneck, 7,
            num_epochs=args.epochs, output_folder=folder, round_n=args.round_n, c=args.c, s=args.s, structure=args.structure)
    compute_convexities(folder, args.round_n)
