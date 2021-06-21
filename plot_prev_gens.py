
'''
Creates the plot of the 5 bottlenecks after the iterated learning is finished
'''

import pickle
import matplotlib.pyplot as plt
import partition
import argparse
import numpy as np
import os


def plot_agents_convexities_one_fig(round_n, epoch, number_of_trials, c, s):

    files = 6
    fig = plt.figure()

    iter_01 = "Different_intis/c1.0_s0.0/trial_1.0_r1_epoch6_c1.0_s0.0"

    iter_02 = "Different_intis/c1.0_s1.0/trial_1.0_r1_epoch6_c1.0_s1.0"

    iter_03 = "high_connected"

    names = ["(a) Random structure, Low inital convexity",
             "(b) Random structure, High inital convexity",
             "(c) Structured initial color space"]
    iters = [iter_01, iter_02, iter_03]
    # 0.72838543593391947

    fig = plt.figure(figsize=plt.figaspect(0.9))

    for x, iter in enumerate(iters):

        average = 0

        ax = fig.add_subplot(3, 1, x+1)
        for i in range(1, 6):

            z = i
            if x == 2:
                z = i + ((x * 5) - 5)
            with open(iter + '/prev_convexities/prev_convexities_' + str(z) + '.pkl', 'rb') as f:
                convexities_02 = pickle.load(f)

            generations = [*range(0, len(convexities_02))]

            ax.plot(generations[:50], convexities_02[:50], label=str(z))

            ax.set_title(names[x])
            ax.set_ylim([0, 1.1])

            ax.set_xlabel('Generations', loc='right', labelpad=15, fontsize=13)
            print("x")
            average += np.mean(convexities_02[:50])

        print(average/5)

    ax.set_ylabel('Degree of convexity', loc='center',
                  labelpad=20, fontsize=13)

    fig.suptitle('Convexities over ' + str(len(generations)) +
                 ' generations, epochs = ' + str(epoch) + ', rounds = 5')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--round_n", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--s", type=float, default=0.0)
    args = parser.parse_args()
    plot_agents_convexities_one_fig(
        args.round_n, args.epochs, args.n_trials, c=args.c, s=args.s)
