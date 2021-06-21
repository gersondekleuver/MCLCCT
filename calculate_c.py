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

    agent = 0
    steps = 50

    # get keys
    total_score = 0.0

    fig = plt.figure(figsize=plt.figaspect(0.3))
    fig.subplots_adjust(hspace=.5)

    for i, dir in enumerate(dirs):

        ax = fig.add_subplot(1, 3, i+1)
        scores = []

        for s in range(1, steps):
            max_edge = 0
            self_connect = 0

            with open(dir + "/generation_" + str(s) + "_1" + '.pkl', 'rb') as f:
                color_space = pickle.load(f)

            part = color_space.partitions[agent].partition
            points = color_space.partitions[agent].points

            for label in part:
                # loop over colors of label
                color_cat = part[label]

                for pt in color_cat:

                    # self
                    origin = points[pt]

                    max_size = points.shape[0]
                    x_offset = 1
                    y_offset = round(max_size**(1/3))
                    z_offset = y_offset * y_offset

                    if pt - x_offset >= 0:
                        max_edge += 1

                        if pt - x_offset in color_cat:
                            self_connect += 1

                    if pt + x_offset <= max_size - 1:
                        max_edge += 1
                        if pt + x_offset in color_cat:
                            self_connect += 1

                    if pt + y_offset <= max_size - 1:
                        max_edge += 1
                        if pt + y_offset in color_cat:
                            self_connect += 1

                    if pt - y_offset >= 0:
                        max_edge += 1
                        if pt - y_offset in color_cat:
                            self_connect += 1

                    if pt + z_offset <= max_size - 1:
                        max_edge += 1
                        if pt + z_offset in color_cat:
                            self_connect += 1

                    if pt - z_offset >= 0:
                        max_edge += 1
                        if pt - z_offset in color_cat:
                            self_connect += 1

            scores.append(self_connect / max_edge)
        ax.plot(np.arange(50), scores)
        print("connectedness", np.mean(scores))
        ax.set_ylabel("Connectedness")
        ax.set_xlabel("Generations")

        ax.set_ylim([0, 1])

    plt.show()


if __name__ == '__main__':

    plot_3D()
