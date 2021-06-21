from __future__ import division
import pickle
import matplotlib.pyplot as plt
import partition
import argparse
import numpy as np
import os
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.utils.extmath import cartesian


def plot_3D():

    agents = 10
    steps = 50

    # get keys
    total_score = 0.0

    fig = plt.figure(figsize=plt.figaspect(0.3))
    fig.subplots_adjust(hspace=.5)

    for i, dir in enumerate(directories):

        ax = fig.add_subplot(1, 3, i+1)
        scores = []

        for s in range(0, steps):
            max_edge = 0
            self_connect = 0
            agent_score = 0

            with open(dir + "/generation_" + str(s) + "_1" + '.pkl', 'rb') as f:
                color_space = pickle.load(f)

            for agent in range(agents):
                part = color_space.partitions[agent].partition
                points = color_space.partitions[agent].points
                labels = 0
                for label in part:
                    # loop over colors of label

                    color_cat = part[label]

                    if len(color_cat) >= 278:
                        labels += 1
                agent_score += labels

            scores.append(agent_score / agents)

        ax.plot(np.arange(50), scores)
        print(np.mean(scores))

        ax.set_ylabel("Average color terms")
        ax.set_xlabel("Generations")
        ax.set_ylim(0, 7)

    plt.show()


if __name__ == '__main__':

    plot_3D()
